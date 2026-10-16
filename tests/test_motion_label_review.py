import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen


spec = importlib.util.spec_from_file_location(
    "review_motion_labels", Path(__file__).resolve().parents[1] / "scripts/review_motion_labels.py"
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class MotionLabelReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "jump.mp4").write_bytes(b"0123456789")
        (self.root / "jump_compressed.mp4").write_bytes(b"duplicate")
        self.caption = self.root / "jump.json"
        self.caption.write_text(json.dumps(dict(summary="Generated", labels=["jump"], events=[], duration_s=20)))
        self.original = self.caption.read_bytes()
        self.review_file = self.root / "edits/reviews.json"
        self.store = review.ReviewStore(self.root, self.review_file)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), review.make_handler(self.store))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def test_save_reload_preserves_generated_caption(self):
        with urlopen(self.url + "/api/clips") as response:
            self.assertEqual(json.load(response), [dict(id="jump", reviewed=False)])
        with urlopen(self.url + "/api/clip/jump") as response:
            self.assertEqual(json.load(response)["summary"], "Generated")
        payload = dict(summary="Human correction", labels=["hop", "land"], reviewed=True,
                       events=[dict(start_s=2, end_s=4, label="hop", description="A short hop")])
        request = Request(self.url + "/api/clip/jump", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            self.assertEqual(response.status, 200)
        restarted = review.ReviewStore(self.root, self.review_file)
        self.assertEqual(restarted.clip("jump")["summary"], "Human correction")
        self.assertTrue(restarted.clip("jump")["reviewed"])
        self.assertEqual(restarted.clip("jump")["events"], payload["events"])
        self.assertEqual(self.caption.read_bytes(), self.original)

    def test_video_seeking_and_page(self):
        with urlopen(self.url) as response:
            self.assertIn(b'<video id="video"', response.read())
        for header, expected in [("bytes=3-6", b"3456"), ("bytes=7-", b"789"), ("bytes=-2", b"89")]:
            with urlopen(Request(self.url + "/video/jump", headers={"Range": header})) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), expected)
        with self.assertRaises(HTTPError) as context:
            urlopen(Request(self.url + "/video/jump", headers={"Range": "bytes=100-200"}))
        self.assertEqual(context.exception.code, 416)
        with self.assertRaises(HTTPError) as context:
            urlopen(self.url + "/video/jump_compressed")
        self.assertEqual(context.exception.code, 404)

    def test_bad_edit_does_not_replace_saved_data(self):
        payload = dict(summary="Corrected", labels=[], reviewed=False, events=[])
        self.store.save("jump", payload)
        saved = self.review_file.read_bytes()
        payload["events"] = [dict(start_s=5, end_s=2, label="bad", description="")]
        request = Request(self.url + "/api/clip/jump", data=json.dumps(payload).encode())
        with self.assertRaises(HTTPError) as context:
            urlopen(request)
        self.assertEqual(context.exception.code, 400)
        self.assertEqual(self.review_file.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
