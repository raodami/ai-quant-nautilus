import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
from ai_quant_nautilus.cli import main
from ai_quant_nautilus.cli_commands import dispatch
from unittest.mock import patch


class TestCLI:
    def test_main_no_args_shows_help(self):
        """Test that running without args shows help."""
        with patch('sys.argv', ['aqn']):
            result = main()
            assert result == 0

    def test_dispatch_unknown_command(self):
        """Test dispatch with unknown command."""
        args = type('Args', (), {'command': 'unknown'})()
        result = dispatch(args)
        assert result == 1

    def test_dispatch_status_no_registry(self, tmp_path):
        """Test status command with no registry file."""
        args = type('Args', (), {
            'command': 'status',
            'data_dir': tmp_path,
            'registry': tmp_path / 'nonexistent.json',
        })()
        result = dispatch(args)
        assert result == 1
