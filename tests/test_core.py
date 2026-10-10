"""Unit tests for CVLens core algorithms and profiling functions."""
import unittest
import numpy as np

from app import BM25Index, HNSW, KD, chunk_text, l2, pct
from profiles import analyze, find_skills


class TestVectorMath(unittest.TestCase):
    def test_l2_identical(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(l2(v, v), 0.0)

    def test_l2_orthogonal(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        # distance squared = (1-0)^2 + (0-1)^2 = 2.0
        self.assertAlmostEqual(l2(v1, v2), 2.0)

    def test_pct_score(self):
        self.assertEqual(pct(0.0), 100.0)
        self.assertEqual(pct(2.0), 0.0)


class TestChunking(unittest.TestCase):
    def test_chunking_with_candidate_prefix(self):
        text = " ".join([f"word{i}" for i in range(300)])
        chunks = chunk_text(text, candidate_name="Alice Smith")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0].startswith("Candidate: Alice Smith\n"))


class TestBM25Tokenizer(unittest.TestCase):
    def test_tokenizer(self):
        tokens = BM25Index.tokenize("FastAPI, Python 3.11 & Docker-container!")
        self.assertIn("fastapi", tokens)
        self.assertIn("python", tokens)
        self.assertIn("docker-container", tokens)


class TestProfiles(unittest.TestCase):
    def test_skill_extraction(self):
        sample = "Experienced in Python, PyTorch, Docker and Kubernetes."
        skills = find_skills(sample)
        self.assertIn("Python", skills)
        self.assertIn("PyTorch", skills)
        self.assertIn("Docker", skills)
        self.assertIn("Kubernetes", skills)

    def test_analyze_education_and_skills(self):
        sample = """
        EXPERIENCE
        Jan 2021 - Present
        Senior Software Engineer
        Working with Python, FastAPI, and PostgreSQL.

        EDUCATION & CERTIFICATIONS
        Bachelor of Science in Computer Science, 2020
        """
        profile = analyze(sample)
        self.assertEqual(profile["edu_label"], "Bachelor's")
        self.assertIn("Python", profile["skills"])
        self.assertIn("FastAPI", profile["skills"])
        self.assertGreater(profile["years"], 0.0)


if __name__ == "__main__":
    unittest.main()
