"""Unit tests for the pure decision logic behind notebook cell 3's torchao
self-heal (see scripts/torchao_selfheal.py for the full story).
"""

import pytest

from scripts.torchao_selfheal import (
    MIN_TORCHAO_VERSION,
    decide_action,
    get_installed_version,
    is_version_below,
    parse_version_prefix,
    uninstall_commands,
)


class TestParseVersionPrefix:
    def test_plain_three_part(self):
        assert parse_version_prefix("0.10.0") == (0, 10, 0)

    def test_local_build_suffix(self):
        assert parse_version_prefix("0.10.0+cu121") == (0, 10, 0)

    def test_dev_suffix(self):
        assert parse_version_prefix("0.16.0.dev123") == (0, 16, 0)

    def test_rc_suffix_dash(self):
        assert parse_version_prefix("0.16.0-rc1") == (0, 16, 0)

    def test_two_part(self):
        assert parse_version_prefix("0.9") == (0, 9)

    def test_unparseable(self):
        assert parse_version_prefix("not-a-version") == ()

    def test_empty_string(self):
        assert parse_version_prefix("") == ()


class TestIsVersionBelow:
    def test_below_default_threshold(self):
        assert is_version_below("0.10.0") is True

    def test_at_default_threshold(self):
        assert is_version_below("0.16.0") is False

    def test_above_default_threshold(self):
        assert is_version_below("0.20.1") is False

    def test_just_below_threshold(self):
        assert is_version_below("0.15.99") is True

    def test_string_comparison_would_be_wrong(self):
        # "0.9.0" < "0.16.0" numerically, but "0.9.0" > "0.16.0" as a string -
        # this pins the numeric (not lexicographic) comparison.
        assert is_version_below("0.9.0") is True

    def test_local_build_suffix_at_threshold(self):
        assert is_version_below("0.16.0+cu121") is False

    def test_local_build_suffix_below_threshold(self):
        assert is_version_below("0.10.0+cu121") is True

    def test_unparseable_treated_as_below(self):
        assert is_version_below("garbage") is True

    def test_custom_threshold(self):
        assert is_version_below("1.5.0", threshold=(2, 0)) is True
        assert is_version_below("2.0.0", threshold=(2, 0)) is False

    def test_default_threshold_is_0_16(self):
        assert MIN_TORCHAO_VERSION == (0, 16)


class TestDecideAction:
    def test_absent(self):
        assert decide_action(None) == "skip-absent"

    def test_below_threshold(self):
        assert decide_action("0.10.0") == "uninstall"

    def test_at_or_above_threshold(self):
        assert decide_action("0.16.0") == "skip-ok"
        assert decide_action("1.0.0") == "skip-ok"


class TestGetInstalledVersion:
    def test_absent_package_returns_none(self):
        assert get_installed_version("this-package-does-not-exist-xyz") is None

    def test_present_package_returns_a_string(self):
        # pytest itself is always installed in this environment.
        result = get_installed_version("pytest")
        assert isinstance(result, str)
        assert result != ""


@pytest.mark.parametrize(
    "installed,expected_action",
    [
        (None, "skip-absent"),
        ("0.0.1", "uninstall"),
        ("0.10.0", "uninstall"),
        ("0.15.99", "uninstall"),
        ("0.16.0", "skip-ok"),
        ("0.16.0+cu121", "skip-ok"),
        ("1.0.0", "skip-ok"),
    ],
)
def test_end_to_end_decision_table(installed, expected_action):
    assert decide_action(installed) == expected_action


# --- uninstall_commands --------------------------------------------------
#
# The first version of this self-heal trusted `pip uninstall`'s return code
# and printed "torchao removed" on exit 0. pip exits 0 when the package is
# not installed in the environment that pip manages, so on a Colab kernel
# whose sys.executable does not own /usr/local/lib/.../dist-packages the
# cell reported success, and the model load died three minutes later with
# the exact ImportError the self-heal exists to prevent.


def test_offers_a_fallback_to_the_pip_on_path():
    """sys.executable's pip and the `pip` on PATH are not always the same
    environment on Colab. Trying only the first one is how a removal that
    never happened gets reported as a success."""
    commands = uninstall_commands("/usr/bin/python3")
    assert commands[0][:3] == ["/usr/bin/python3", "-m", "pip"]
    assert commands[1][0] == "pip"


def test_every_command_is_a_non_interactive_torchao_uninstall():
    for command in uninstall_commands("/usr/bin/python3"):
        assert "uninstall" in command
        assert "-y" in command, "a prompt would hang the notebook cell"
        assert command[-1] == "torchao"


def test_get_installed_version_reports_absent_packages_as_none():
    assert get_installed_version("a-package-that-is-not-installed-anywhere") is None


def test_get_installed_version_finds_a_package_that_is_installed():
    """Guards the None-means-absent contract against a silently broken
    lookup, which would make the self-heal a no-op that always claims
    success."""
    assert get_installed_version("pytest") is not None
