"""Tests for capability-based PaperContext views."""

import unittest

from src.domain.paper import Paper
from src.domain.paper_context import PaperContext


class TestContextPermissions(unittest.TestCase):
    def test_mutable_field_is_not_exposed_as_attribute(self):
        context = PaperContext(sections={"intro": "draft"})
        view = context.view(readable={"sections"}, writable=set())
        with self.assertRaises(AttributeError):
            _ = view.sections

    def test_read_returns_detached_immutable_snapshot(self):
        context = PaperContext(seed_pdf_paths=["a.pdf"])
        view = context.view(readable={"seed_pdf_paths"}, writable=set())
        snapshot = view.read("seed_pdf_paths")
        self.assertEqual(snapshot, ("a.pdf",))
        self.assertIsInstance(snapshot, tuple)

    def test_domain_command_enforces_write_permission(self):
        paper = Paper(cite_key="key", title="Title", authors="A", year=2026)
        context = PaperContext()
        readonly = context.view(readable={"reference_library"}, writable=set())
        with self.assertRaises(AttributeError):
            readonly.add_reference(paper)
        writer = context.view(
            readable={"reference_library"}, writable={"reference_library"})
        writer.add_reference(paper)
        self.assertEqual(writer.get_references()[0].cite_key, "key")

    def test_reference_snapshot_does_not_mutate_source(self):
        paper = Paper(cite_key="key", title="Original", authors="A", year=2026)
        context = PaperContext(reference_library=[paper])
        view = context.view(readable={"reference_library"}, writable=set())
        snapshot = view.get_references()
        snapshot[0].title = "Changed"
        self.assertEqual(context.reference_library[0].title, "Original")

    def test_main_tex_path_uses_authorized_command(self):
        context = PaperContext()
        writer = context.view(readable={"main_tex_path"}, writable={"main_tex_path"})
        writer.set_main_tex_path("output/main.tex")
        self.assertEqual(context.main_tex_path, "output/main.tex")


if __name__ == "__main__":
    unittest.main()
