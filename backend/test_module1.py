"""
Unit and Integration Test Suite for Module 1: Meat Intake & Predictive Shelf Life.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database as db
from shelf_life_engine import predict_shelf_life


class TestModule1(unittest.TestCase):
    def setUp(self):
        # Use an isolated test database
        self.temp_dir = tempfile.mkdtemp()
        self.orig_db_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir) / "test_marbl.db"
        db.init_db()

    def tearDown(self):
        db.DB_PATH = self.orig_db_path

    def test_predict_shelf_life_kinetics(self):
        # 1. Normal Beef Vacuum at 1.5°C
        p_beef_vac = predict_shelf_life(
            meat_type="beef",
            initial_ph=5.60,
            receiving_temp_c=1.5,
            packaging_type="vacuum",
        )
        self.assertGreater(p_beef_vac.shelf_life_days, 20.0)
        self.assertLess(p_beef_vac.shelf_life_days, 45.0)
        self.assertEqual(len([w for w in p_beef_vac.warnings if "Cold Chain Breach" in w]), 0)

        # 2. Temperature Abused Chicken in Air
        p_chick_abuse = predict_shelf_life(
            meat_type="chicken",
            initial_ph=6.30,
            receiving_temp_c=6.5,
            transport_excursion=True,
            packaging_type="air",
        )
        self.assertLess(p_chick_abuse.shelf_life_days, 3.0)
        # Should flag cold chain breach
        breach_warnings = [w for w in p_chick_abuse.warnings if "Cold Chain Breach" in w]
        self.assertEqual(len(breach_warnings), 1)

        # 3. High pH DFD Beef
        p_dfd = predict_shelf_life(
            meat_type="beef",
            initial_ph=6.15,
            receiving_temp_c=2.0,
            packaging_type="air",
        )
        dfd_warnings = [w for w in p_dfd.warnings if "High pH" in w or "DFD" in w]
        self.assertEqual(len(dfd_warnings), 1)

        # 4. Pork MAP
        p_pork_map = predict_shelf_life(
            meat_type="pork",
            initial_ph=5.65,
            receiving_temp_c=2.0,
            packaging_type="map",
        )
        self.assertGreater(p_pork_map.shelf_life_days, 8.0)
        self.assertLess(p_pork_map.shelf_life_days, 20.0)

    def test_database_crud(self):
        batch_id = db.generate_batch_id()
        self.assertTrue(batch_id.startswith("MB-"))

        # Predict shelf life first
        p = predict_shelf_life("beef", 5.6, 2.0, packaging_type="vacuum")

        batch_data = {
            "batch_id": batch_id,
            "meat_type": "beef",
            "initial_ph": 5.60,
            "receiving_temp_c": 2.0,
            "time_since_slaughter_hrs": 24.0,
            "transport_temp_avg_c": 1.8,
            "transport_duration_hrs": 5.0,
            "transport_excursion": False,
            "packaging_type": "vacuum",
            "initial_weight_val": 45.2,
            "initial_weight_unit": "kg",
            "shelf_life_days": p.shelf_life_days,
            "shelf_life_min_days": p.shelf_life_min_days,
            "shelf_life_max_days": p.shelf_life_max_days,
            "confidence_score": p.confidence_score,
            "recommended_temp_c": p.recommended_temp_c,
            "warnings": p.warnings,
        }

        created = db.create_batch(batch_data)
        self.assertEqual(created["batch_id"], batch_id)
        self.assertEqual(created["initial_weight_val"], 45.2)
        self.assertEqual(created["initial_weight_unit"], "kg")

        # Staged drip loss test was created
        self.assertIsNotNone(created["drip_loss_staged"])
        self.assertEqual(created["drip_loss_staged"]["test_status"], "in_development")

        # Add visual inspection record
        rec_id = db.add_inspection_record(batch_id, {
            "filename": "sample1.jpg",
            "decision": "packing",
            "good_confidence": 0.94,
            "spoiled_confidence": 0.06,
            "size_category": "big",
            "longest_dimension_mm": 135.0,
            "piece_count_estimate": 1,
            "fat_coverage_pct": 8.7,
            "lean_coverage_pct": 91.3,
            "fat_category": "low",
            "segmentation_quality": 0.82,
        })
        self.assertGreater(rec_id, 0)

        # Retrieve batch again and check stats
        retrieved = db.get_batch(batch_id)
        self.assertEqual(retrieved["total_inspections"], 1)
        self.assertEqual(retrieved["passed_count"], 1)
        self.assertEqual(retrieved["discarded_count"], 0)
        self.assertEqual(retrieved["avg_fat_pct"], 8.7)

        # List batches
        batches = db.list_batches(10)
        self.assertTrue(any(b["batch_id"] == batch_id for b in batches))


if __name__ == "__main__":
    unittest.main()
