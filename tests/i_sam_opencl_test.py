# SPDX-License-Identifier: Unlicense
"""Tests for i.sam.opencl."""

import functools
import http.server
import shutil
import threading
import zipfile
from pathlib import Path

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


def _tools_with_env(session, **env):
    return Tools(env={**session.env, **env})


def test_default_weights_dir(session, tmp_path):
    """Without checkpoint or weights_dir, the central folder
    $SAM_MODELS_DIR/sam1 is searched."""
    tools = _tools_with_env(session, SAM_MODELS_DIR=str(tmp_path))
    expected = tmp_path / "sam1"

    with pytest.raises(ToolError, match=r"No\s+SAM\s+checkpoint") as e:
        tools.i_sam_opencl(group="rgb", output="out", flags="n")
    assert str(expected) in " ".join(str(e.value).split())


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


def test_model_excludes_checkpoint(session, tmp_path):
    """model= chooses among the files of a weights folder only."""
    tools = Tools(session=session)
    dummy = tmp_path / "dummy.pth"
    dummy.write_bytes(b"not used")

    with pytest.raises(ToolError, match=r"mutually\s+exclusive"):
        tools.i_sam_opencl(
            group="rgb", checkpoint=str(dummy), model="vit_b", output="out"
        )


def test_empty_weights_dir(session, tmp_path):
    """A folder without SAM checkpoints fails before any OpenCL work."""
    tools = Tools(session=session)
    (tmp_path / "readme.txt").write_text("no weights here")

    with pytest.raises(ToolError, match=r"No\s+SAM\s+checkpoint"):
        tools.i_sam_opencl(
            group="rgb", weights_dir=str(tmp_path), output="out", flags="n"
        )


def test_weights_dir_without_requested_model(session, tmp_path):
    """Asking for a model that is not in the folder is an error, not a
    silent fallback to another model."""
    tools = Tools(session=session)
    (tmp_path / "sam_vit_h_4b8939.pth").write_bytes(b"not used")

    with pytest.raises(ToolError, match=r"No\s+vit_b\s+checkpoint"):
        tools.i_sam_opencl(
            group="rgb",
            weights_dir=str(tmp_path),
            model="vit_b",
            output="out",
            flags="n",
        )


def test_weights_dir_checks_model_of_file(session, checkpoint, tmp_path):
    """A file named as another model than the one it holds is refused."""
    tools = Tools(session=session)
    name = "vit_h" if "vit_b" in Path(checkpoint).name else "vit_b"
    (tmp_path / f"sam_{name}_renamed.pth").symlink_to(checkpoint)

    with pytest.raises(ToolError, match=r"does\s+not\s+hold"):
        tools.i_sam_opencl(
            group="rgb", weights_dir=str(tmp_path), model=name, output="out"
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


# Published sizes of the checkpoints the module downloads.
RELEASES = {
    "vit_h": ("sam_vit_h_4b8939.pth", 2564550879),
    "vit_l": ("sam_vit_l_0b3195.pth", 1249524607),
    "vit_b": ("sam_vit_b_01ec64.pth", 375042383),
}


@pytest.fixture
def mirror(tmp_path):
    """A local HTTP server standing in for the download site; yields the
    folder it serves and its URL."""
    root = tmp_path / "mirror"
    root.mkdir()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_download_rejects_wrong_size(session, tmp_path, mirror):
    """A download that does not have the published size is discarded, not
    installed in the weights folder."""
    root, url = mirror
    (root / "sam_vit_b_01ec64.pth").write_bytes(b"truncated")
    weights = tmp_path / "weights"
    tools = _tools_with_env(session, I_SAM_OPENCL_DOWNLOAD_URL=url)

    with pytest.raises(ToolError, match=r"instead\s+of\s+375042383"):
        tools.i_sam_opencl(
            group="rgb", weights_dir=str(weights), model="vit_b", output="out"
        )
    assert not list(weights.glob("*.pth*"))


def test_download_fetches_missing_model(session, checkpoint, tmp_path, mirror):
    """A model missing from the weights folder (created if needed) is
    downloaded there and used."""
    variant = next(
        (
            v
            for v, (_, size) in RELEASES.items()
            if Path(checkpoint).stat().st_size == size
        ),
        None,
    )
    if variant is None:
        pytest.skip("checkpoint is not one of the published release files")
    name, size = RELEASES[variant]
    if shutil.disk_usage(tmp_path).free < 2 * size:
        pytest.skip(f"not enough free space in {tmp_path} for a {size} byte copy")
    root, url = mirror
    (root / name).symlink_to(checkpoint)
    weights = tmp_path / "new" / "weights"
    tools = _tools_with_env(session, I_SAM_OPENCL_DOWNLOAD_URL=url)

    # A platform that does not exist stops the run right after the
    # checkpoint has been fetched and read.
    with pytest.raises(ToolError, match=r"OpenCL") as e:
        tools.i_sam_opencl(
            group="rgb",
            weights_dir=str(weights),
            model=variant,
            output="out",
            platform="no_such_platform",
        )
    assert "does not hold" not in str(e.value)
    assert (weights / name).stat().st_size == size
    assert not list(weights.glob("*.part"))


@pytest.mark.parametrize("option", ["profile", "precision"])
def test_rejects_invalid_device_option(session, tmp_path, option):
    """profile= and precision= only take their listed values."""
    tools = Tools(session=session)
    with pytest.raises(ToolError):
        tools.i_sam_opencl(
            group="rgb",
            checkpoint=str(tmp_path / "missing.pt"),
            output="out",
            **{option: "nonsense"},
        )


@pytest.mark.parametrize(
    ("profile", "precision"), [("cpu", "fp32"), ("igpu", "fp32"), ("dgpu", "fp16")]
)
def test_profiles_and_precision_give_the_rectangles(
    session, checkpoint, device_options, profile, precision
):
    """A forced device profile, and half-precision weights, still recover
    every rectangle at the right size."""
    tools = Tools(session=session)
    out = f"seg_{profile}_{precision}"
    tools.i_sam_opencl(
        group="rgb",
        checkpoint=checkpoint,
        output=out,
        raster=out,
        points_per_side=8,
        minsize=50,
        profile=profile,
        precision=precision,
        **device_options,
    )
    for name, n, s, w, e, _ in RECTANGLES:
        x, y = (w + e) // 2 - 0.5, 256 - (n + s) // 2 + 0.5
        cat = tools.r_what(map=out, coordinates=(x, y), format="json").json[0][out][
            "value"
        ]
        assert cat is not None, f"{name} rectangle not segmented"
        row = tools.v_db_select(map=out, where=f"cat = {cat}", format="json").json[
            "records"
        ][0]
        assert row["cells"] == pytest.approx((s - n + 1) * (e - w + 1), rel=0.05)
