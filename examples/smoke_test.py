from pathlib import Path
import tempfile

import torch

from capybara import CAPY, default_config, load_config


def main() -> None:
    model = CAPY(default_config())
    model.eval()

    with torch.no_grad():
        x = torch.randn(2, 4, 2048)
        profile_logits, log_counts = model(x)

    assert profile_logits.shape == (2, 2, 1000), profile_logits.shape
    assert log_counts.shape == (2, 1), log_counts.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "capy_state_dict.pt"
        torch.save(model.state_dict(), checkpoint_path)

        reloaded = CAPY(default_config())
        reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        reloaded.eval()

        with torch.no_grad():
            reloaded_profile_logits, reloaded_log_counts = reloaded(x)

    assert reloaded_profile_logits.shape == (2, 2, 1000), reloaded_profile_logits.shape
    assert reloaded_log_counts.shape == (2, 1), reloaded_log_counts.shape

    yaml_config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    try:
        yaml_model = CAPY(load_config(yaml_config_path))
    except ImportError:
        yaml_model = None
    if yaml_model is not None:
        yaml_model.eval()
        with torch.no_grad():
            yaml_profile_logits, yaml_log_counts = yaml_model(x)
        assert yaml_profile_logits.shape == (2, 2, 1000), yaml_profile_logits.shape
        assert yaml_log_counts.shape == (2, 1), yaml_log_counts.shape

    print("CAPY smoke test passed.")


if __name__ == "__main__":
    main()
