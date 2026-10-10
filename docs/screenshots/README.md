# 📸 UI Screenshots & Visual Assets

This folder hosts the visual previews and screenshots displayed in the root [`README.md`](../../README.md).

## Quick Guide: Uploading Your Screenshots

To update any screenshot on GitHub, simply replace the corresponding image file in this folder with your actual screenshot (PNG format recommended):

| File | Page / Section | Recommended Dimensions | Description |
|---|---|---|---|
| [`01-overview.png`](./01-overview.png) | **Overview Dashboard** | 1920×1080 or 1280×720 | Main dashboard displaying system status, Ollama health, and quick actions |
| [`02-upload-cvs.png`](./02-upload-cvs.png) | **My CVs & Upload** | 1920×1080 or 1280×720 | Drag & drop resume uploader and list of indexed candidate resumes |
| [`03-job-match.png`](./03-job-match.png) | **Job Match & Fit Analysis** | 1920×1080 or 1280×720 | Job description matching, ranking score bars, and LLM explanation |
| [`04-candidate-explorer.png`](./04-candidate-explorer.png) | **Candidate Explorer** | 1920×1080 or 1280×720 | Multi-attribute candidate table with experience, education, and skill filters |
| [`05-ai-assistant.png`](./05-ai-assistant.png) | **AI Recruiting Assistant** | 1920×1080 or 1280×720 | Natural language recruiter Q&A assistant with candidate citations |
| [`06-benchmark.png`](./06-benchmark.png) | **Algorithm Benchmark** | 1920×1080 or 1280×720 | Live benchmark comparing HNSW, KD-Tree, and Brute-force recall and latency |

After replacing any file, commit and push:
```bash
git add docs/screenshots/
git commit -m "docs: update UI screenshots"
git push origin main
```
