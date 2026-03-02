"""Tests for QuantilePreScanner class."""

from datetime import datetime, timedelta

import pytest

from ehr_fm.vocabulary import QuantilePreScanner


class TestQuantilePreScannerInit:
    """Test QuantilePreScanner initialization."""

    def test_default_init(self):
        """Initialize with defaults."""
        scanner = QuantilePreScanner()
        assert scanner.num_quantiles == 10
        assert scanner.reservoir_size == 10000
        assert len(scanner.reservoirs) == 0
        assert len(scanner.discovered_stages) == 0

    def test_custom_quantiles(self):
        """Initialize with custom num_quantiles."""
        scanner = QuantilePreScanner(num_quantiles=5)
        assert scanner.num_quantiles == 5

    def test_custom_reservoir_size(self):
        """Initialize with custom reservoir_size."""
        scanner = QuantilePreScanner(reservoir_size=1000)
        assert scanner.reservoir_size == 1000


class TestQuantilePreScannerForward:
    """Test QuantilePreScanner.forward() sample collection."""

    @pytest.fixture
    def scanner(self):
        """QuantilePreScanner instance."""
        return QuantilePreScanner(num_quantiles=5, reservoir_size=100)

    def test_forward_collects_numeric_values(self, scanner):
        """Forward collects numeric values into reservoirs."""
        birth_time = datetime(2000, 1, 1)
        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=30), "numeric_value": 80.0},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=60), "numeric_value": 120.0},
            ],
        ]
        scanner.forward(batch)

        assert "LAB/glucose" in scanner.reservoirs
        assert scanner.reservoirs["LAB/glucose"].n == 2

    def test_forward_discovers_stages(self, scanner):
        """Forward discovers workflow_stage values."""
        birth_time = datetime(2000, 1, 1)
        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=30),
                    "numeric_value": None,
                    "workflow_stage": "taken",
                },
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=60),
                    "numeric_value": None,
                    "workflow_stage": "order",
                },
            ],
        ]
        scanner.forward(batch)

        assert "taken" in scanner.discovered_stages
        assert "order" in scanner.discovered_stages

    def test_forward_ignores_none_numeric(self, scanner):
        """Forward ignores events with None numeric_value."""
        birth_time = datetime(2000, 1, 1)
        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=30), "numeric_value": None},
            ],
        ]
        scanner.forward(batch)

        # No reservoir created since no numeric values
        assert "LAB/glucose" not in scanner.reservoirs

    def test_forward_handles_multiple_batches(self, scanner):
        """Forward accumulates across multiple batches."""
        birth_time = datetime(2000, 1, 1)
        batch1 = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=30), "numeric_value": 80.0},
            ],
        ]
        batch2 = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=30), "numeric_value": 120.0},
            ],
        ]
        scanner.forward(batch1)
        scanner.forward(batch2)

        assert scanner.reservoirs["LAB/glucose"].n == 2

    def test_forward_handles_empty_batch(self, scanner):
        """Forward handles empty batch gracefully."""
        scanner.forward([[]])
        assert len(scanner.reservoirs) == 0


class TestQuantilePreScannerComputeBreaks:
    """Test QuantilePreScanner.compute_breaks() break point calculation."""

    def test_compute_breaks_basic(self):
        """Compute breaks with simple values."""
        scanner = QuantilePreScanner(num_quantiles=3)  # 2 break points
        birth_time = datetime(2000, 1, 1)

        # Add values that span a range
        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=1), "numeric_value": 60.0},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=2), "numeric_value": 80.0},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=3), "numeric_value": 100.0},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=4), "numeric_value": 120.0},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=5), "numeric_value": 140.0},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=6), "numeric_value": 160.0},
            ],
        ]
        scanner.forward(batch)
        breaks = scanner.compute_breaks()

        assert "LAB/glucose" in breaks
        # With 3 quantiles, should have 2 break points
        assert len(breaks["LAB/glucose"]) == 2

    def test_compute_breaks_invariant_values(self):
        """Invariant values (all same) → empty breaks list."""
        scanner = QuantilePreScanner(num_quantiles=5)
        birth_time = datetime(2000, 1, 1)

        # All values identical
        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=1), "numeric_value": 100.0},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=2), "numeric_value": 100.0},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=3), "numeric_value": 100.0},
            ],
        ]
        scanner.forward(batch)
        breaks = scanner.compute_breaks()

        # Invariant values → empty breaks list
        assert breaks["LAB/glucose"] == []

    def test_compute_breaks_multiple_codes(self):
        """Compute breaks for multiple codes independently."""
        scanner = QuantilePreScanner(num_quantiles=3)
        birth_time = datetime(2000, 1, 1)

        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=1), "numeric_value": 80.0},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=2), "numeric_value": 120.0},
                {"code": "LAB/hba1c", "time": birth_time + timedelta(days=3), "numeric_value": 5.0},
                {"code": "LAB/hba1c", "time": birth_time + timedelta(days=4), "numeric_value": 8.0},
            ],
        ]
        scanner.forward(batch)
        breaks = scanner.compute_breaks()

        assert "LAB/glucose" in breaks
        assert "LAB/hba1c" in breaks

    def test_compute_breaks_deduplicates_consecutive(self):
        """Consecutive identical break values are deduplicated."""
        scanner = QuantilePreScanner(num_quantiles=5)
        birth_time = datetime(2000, 1, 1)

        # Values that would produce duplicate breaks
        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {"code": "LAB/test", "time": birth_time + timedelta(days=1), "numeric_value": 0.0},
                {"code": "LAB/test", "time": birth_time + timedelta(days=2), "numeric_value": 0.0},
                {"code": "LAB/test", "time": birth_time + timedelta(days=3), "numeric_value": 0.0},
                {"code": "LAB/test", "time": birth_time + timedelta(days=4), "numeric_value": 100.0},
            ],
        ]
        scanner.forward(batch)
        breaks = scanner.compute_breaks()

        # Should not have duplicate 0.0 breaks
        break_values = breaks["LAB/test"]
        assert len(break_values) == len(set(break_values))

    def test_compute_breaks_empty_scanner(self):
        """Empty scanner returns empty breaks dict."""
        scanner = QuantilePreScanner()
        breaks = scanner.compute_breaks()
        assert breaks == {}


class TestQuantilePreScannerGetDiscoveredStages:
    """Test QuantilePreScanner.get_discovered_stages() stage discovery."""

    def test_get_discovered_stages_basic(self):
        """Discover stages from workflow_stage field."""
        scanner = QuantilePreScanner()
        birth_time = datetime(2000, 1, 1)

        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=1),
                    "numeric_value": None,
                    "workflow_stage": "taken",
                },
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=2),
                    "numeric_value": None,
                    "workflow_stage": "order",
                },
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=3),
                    "numeric_value": None,
                    "workflow_stage": "admin",
                },
            ],
        ]
        scanner.forward(batch)
        stages = scanner.get_discovered_stages()

        assert set(stages) == {"taken", "order", "admin"}

    def test_get_discovered_stages_sorted(self):
        """Discovered stages returned sorted."""
        scanner = QuantilePreScanner()
        birth_time = datetime(2000, 1, 1)

        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=1),
                    "numeric_value": None,
                    "workflow_stage": "zebra",
                },
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=2),
                    "numeric_value": None,
                    "workflow_stage": "alpha",
                },
            ],
        ]
        scanner.forward(batch)
        stages = scanner.get_discovered_stages()

        assert stages == sorted(stages)

    def test_get_discovered_stages_lowercase(self):
        """Stages stored as lowercase."""
        scanner = QuantilePreScanner()
        birth_time = datetime(2000, 1, 1)

        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=1),
                    "numeric_value": None,
                    "workflow_stage": "TAKEN",
                },
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=2),
                    "numeric_value": None,
                    "workflow_stage": "Order",
                },
            ],
        ]
        scanner.forward(batch)
        stages = scanner.get_discovered_stages()

        assert "taken" in stages
        assert "order" in stages
        assert "TAKEN" not in stages

    def test_get_discovered_stages_deduplicated(self):
        """Duplicate stages only appear once."""
        scanner = QuantilePreScanner()
        birth_time = datetime(2000, 1, 1)

        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=1),
                    "numeric_value": None,
                    "workflow_stage": "taken",
                },
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=2),
                    "numeric_value": None,
                    "workflow_stage": "taken",
                },
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=3),
                    "numeric_value": None,
                    "workflow_stage": "taken",
                },
            ],
        ]
        scanner.forward(batch)
        stages = scanner.get_discovered_stages()

        assert stages == ["taken"]

    def test_get_discovered_stages_empty(self):
        """No stages discovered returns empty list."""
        scanner = QuantilePreScanner()
        birth_time = datetime(2000, 1, 1)

        batch = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=1), "numeric_value": None},
            ],
        ]
        scanner.forward(batch)
        stages = scanner.get_discovered_stages()

        assert stages == []


class TestQuantilePreScannerReservoirSampling:
    """Test reservoir sampling behavior for large datasets."""

    def test_reservoir_respects_size_limit(self):
        """Reservoir samples don't exceed size limit."""
        scanner = QuantilePreScanner(reservoir_size=5)
        birth_time = datetime(2000, 1, 1)

        # Add more values than reservoir size
        events = [{"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None}]
        for i in range(10):
            events.append(
                {"code": "LAB/test", "time": birth_time + timedelta(days=i + 1), "numeric_value": float(i)}
            )

        scanner.forward([events])

        # Reservoir should have at most reservoir_size samples
        reservoir = scanner.reservoirs["LAB/test"]
        assert reservoir.n <= scanner.reservoir_size or len(reservoir.samples) <= scanner.reservoir_size

    def test_reservoir_tracks_total_count(self):
        """Reservoir tracks total samples seen via total_weight."""
        scanner = QuantilePreScanner(reservoir_size=5)
        birth_time = datetime(2000, 1, 1)

        events = [{"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None}]
        for i in range(10):
            events.append(
                {"code": "LAB/test", "time": birth_time + timedelta(days=i + 1), "numeric_value": float(i)}
            )

        scanner.forward([events])

        # total_weight tracks total seen (10), n tracks filled slots
        reservoir = scanner.reservoirs["LAB/test"]
        assert reservoir.total_weight == 10.0
        assert reservoir.n <= scanner.reservoir_size


class TestQuantilePreScannerIntegration:
    """Integration tests for QuantilePreScanner workflow."""

    def test_full_prescan_workflow(self):
        """Complete pre-scan workflow produces usable results."""
        scanner = QuantilePreScanner(num_quantiles=4)
        birth_time = datetime(2000, 1, 1)

        # Simulate multiple patient batches
        batch1 = [
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=30),
                    "numeric_value": 70.0,
                    "workflow_stage": "taken",
                },
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=60), "numeric_value": 90.0},
            ],
            [
                {"code": "MEDS_BIRTH", "time": birth_time, "numeric_value": None},
                {
                    "code": "LAB/glucose",
                    "time": birth_time + timedelta(days=30),
                    "numeric_value": 110.0,
                    "workflow_stage": "order",
                },
                {"code": "LAB/glucose", "time": birth_time + timedelta(days=60), "numeric_value": 130.0},
            ],
        ]

        scanner.forward(batch1)

        # Get results
        breaks = scanner.compute_breaks()
        stages = scanner.get_discovered_stages()

        # Verify quantile breaks exist
        assert "LAB/glucose" in breaks
        assert len(breaks["LAB/glucose"]) > 0

        # Verify breaks are in sorted order
        assert breaks["LAB/glucose"] == sorted(breaks["LAB/glucose"])

        # Verify stages discovered
        assert set(stages) == {"order", "taken"}

        # Breaks should divide the value range
        all_breaks = breaks["LAB/glucose"]
        assert all(70.0 <= b <= 130.0 for b in all_breaks)
