"""Tests for i.sam.opencl."""

import zipfile

import pytest
from conftest import RECTANGLES
from grass.tools import ToolError, Tools


def test_group_must_hold_three_bands(session, tmp_path):
    """A group with other than three raster maps is rejected up front."""
    tools = Tools(session=session)
    dummy = tmp_path / "dummy.pth"
    dummy.write_bytes(b"not used")

    with pytest.raises(ToolError, match="exactly 3"):
        tools.i_sam_opencl(group="two", checkpoint=str(dummy), output="out")


def test_rejects_file_that_is_not_a_checkpoint(session, tmp_path):
    """A file that is not a PyTorch zip archive fails before any OpenCL
    work."""
    tools = Tools(session=session)
    bogus = tmp_path / "bogus.pth"
    bogus.write_bytes(b"\0" * 4096)

    with pytest.raises(ToolError, match="not a zip archive"):
        tools.i_sam_opencl(group="rgb", checkpoint=str(bogus), output="out")


def test_rejects_zip_without_state_dict(session, tmp_path):
    """A zip archive without a pickled state_dict is not a checkpoint."""
    tools = Tools(session=session)
    archive = tmp_path / "other.pth"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("other/readme.txt", "hello")

    with pytest.raises(ToolError, match="not a PyTorch checkpoint"):
        tools.i_sam_opencl(group="rgb", checkpoint=str(archive), output="out")


@pytest.fixture(scope="module")
def segmented(session, checkpoint, device_options):
    """Segment the rectangles once for all result checks."""
    tools = Tools(session=session)
    tools.i_sam_opencl(
        group="rgb",
        checkpoint=checkpoint,
        output="segments",
        raster="segments",
        points_per_side=8,
        minsize=50,
        **device_options,
    )
    return tools


def _category_at(tools, row, col):
    """Segment category of the cell at 1-based (row, col)."""
    x, y = col - 0.5, 256 - row + 0.5
    value = tools.r_what(map="segments", coordinates=(x, y), format="json").json
    return value[0]["segments"]["value"]


def test_each_rectangle_is_one_segment(segmented):
    """Every rectangle is recovered as its own segment of the right size."""
    categories = set()
    for name, n, s, w, e, _ in RECTANGLES:
        cat = _category_at(segmented, (n + s) // 2, (w + e) // 2)
        assert cat is not None, f"{name} rectangle not segmented"
        categories.add(cat)

        cells = (s - n + 1) * (e - w + 1)
        row = segmented.v_db_select(
            map="segments", where=f"cat = {cat}", format="json"
        ).json["records"][0]
        assert row["cells"] == pytest.approx(cells, rel=0.05), name
        assert row["area"] == pytest.approx(cells, rel=0.05), name
        assert 0 < row["predicted_iou"] <= 1.1
        assert 0 < row["stability"] <= 1
    assert len(categories) == len(RECTANGLES)


def test_segments_form_a_partition(segmented):
    """Segments do not overlap: the attribute cell counts add up to the
    labelled cells, and every segment has exactly one area."""
    records = segmented.v_db_select(map="segments", format="json").json["records"]
    labelled = segmented.r_univar(map="segments", format="json").json["n"]
    assert sum(r["cells"] for r in records) == labelled

    topo = segmented.v_info(map="segments", flags="t", format="json").json
    assert topo["centroids"] == len(records)
    assert topo["areas"] >= topo["centroids"]


def test_checkpoint_or_weights_dir_is_required(session):
    """One source of weights must be given."""
    tools = Tools(session=session)

    with pytest.raises(ToolError, match=r"At\s+least\s+one"):
        tools.i_sam_opencl(group="rgb", output="out")


def test_checkpoint_and_weights_dir_are_exclusive(session, tmp_path):
    """A checkpoint file and a weights folder cannot both be given."""
    tools = Tools(session=session)
    dummy = tmp_path / "sam_vit_b.pth"
    dummy.write_bytes(b"not used")

    with pytest.raises(ToolError, match=r"mutually\s+exclusive"):
        tools.i_sam_opencl(
            group="rgb",
            checkpoint=str(dummy),
            weights_dir=str(tmp_path),
            output="out",
        )


def test_model_needs_weights_dir(session, tmp_path):
    """model= chooses among the files of a weights folder only."""
    tools = Tools(session=session)
    dummy = tmp_path / "dummy.pth"
    dummy.write_bytes(b"not used")

    with pytest.raises(ToolError, match=r"requires"):
        tools.i_sam_opencl(
            group="rgb", checkpoint=str(dummy), model="vit_b", output="out"
        )


def test_empty_weights_dir(session, tmp_path):
    """A folder without SAM checkpoints fails before any OpenCL work."""
    tools = Tools(session=session)
    (tmp_path / "readme.txt").write_text("no weights here")

    with pytest.raises(ToolError, match=r"No\s+SAM\s+checkpoint"):
        tools.i_sam_opencl(group="rgb", weights_dir=str(tmp_path), output="out")


def test_weights_dir_without_requested_model(session, tmp_path):
    """Asking for a model that is not in the folder is an error, not a
    silent fallback to another model."""
    tools = Tools(session=session)
    (tmp_path / "sam_vit_h_4b8939.pth").write_bytes(b"not used")

    with pytest.raises(ToolError, match=r"No\s+vit_b\s+checkpoint"):
        tools.i_sam_opencl(
            group="rgb", weights_dir=str(tmp_path), model="vit_b", output="out"
        )


def test_weights_dir_checks_model_of_file(session, checkpoint, tmp_path):
    """A file named as another model than the one it holds is refused."""
    tools = Tools(session=session)
    (tmp_path / "sam_vit_b_renamed.pth").symlink_to(checkpoint)

    with pytest.raises(ToolError, match=r"does\s+not\s+hold"):
        tools.i_sam_opencl(
            group="rgb", weights_dir=str(tmp_path), model="vit_b", output="out"
        )


def test_weights_dir_segments(session, checkpoint, device_options, tmp_path):
    """Segmentation runs with the checkpoint found in a weights folder,
    including a renamed copy."""
    tools = Tools(session=session)
    (tmp_path / "sam_vit_h_checkpoint.pth").symlink_to(checkpoint)
    tools.i_sam_opencl(
        group="rgb",
        weights_dir=str(tmp_path),
        output="from_folder",
        points_per_side=8,
        minsize=50,
        **device_options,
    )
    topo = tools.v_info(map="from_folder", flags="t", format="json").json
    assert topo["centroids"] >= len(RECTANGLES)
