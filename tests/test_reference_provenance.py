import json
import unittest
from unittest.mock import patch
from src.domain.reference_provenance import ReferenceProvenance
from src.domain.paper import Paper, Source
from src.domain.paper_context import PaperContext
from src.tools.builtin.citations.storage import AddReferenceTool
from src.tools.builtin.search import VerifyPaperTool, SearchPapersTool
from src.tools.builtin.citations.manage import UpdateReferenceTool


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.gate = ReferenceProvenance()
        self.verify = VerifyPaperTool()
        self.verify.provenance = self.gate
        self.adder = AddReferenceTool()
        self.adder.provenance = self.gate
        self.adder.verifier = self.verify
        self.refs = []
        self.adder.set_add_func(self.refs.append)

    def test_declared_source_cannot_bypass_verification(self):
        for source in ['user','online','llm']:
            with patch.object(self.verify._searcher,'_search_auto',return_value='{"ok":true,"results":[]}') as search:
                result = self.adder.execute('a','Invented','Author',2000,source=source)
            self.assertTrue(result.startswith('Error:'))
            search.assert_called_once()
        self.assertEqual(self.refs,[])

    def test_actual_search_result_admitted_without_verifying_again(self):
        searcher=SearchPapersTool(); searcher.provenance=self.gate
        with patch.object(searcher,'_execute_search',return_value='{"ok":true,"results":[{"title":"Known"}]}'):
            searcher.execute('Known')
        with patch.object(self.verify._searcher,'_search_auto') as network:
            self.adder.execute('a','Known','Author',2000)
            self.verify.execute('Known')
        network.assert_not_called()
        self.assertEqual(self.refs[0].source,Source.ONLINE)
        self.assertEqual(self.refs[0].provenance_kind,'search')

    def test_pdf_receipt_skips_network(self):
        self.gate.record('Known','pdf')
        with patch.object(self.verify._searcher,'_search_auto') as network:
            self.adder.execute('a','Known','Author',2000)
            result=json.loads(self.verify.execute('Known'))
        network.assert_not_called()
        self.assertEqual(self.refs[0].source,Source.USER)
        self.assertEqual(result['status'],'not_required')

    def test_model_reference_verified_on_admission(self):
        with patch.object(self.verify._searcher,'_search_auto',return_value='{"ok":true,"results":[{"title":"Known"}]}') as network:
            self.adder.execute('a','Known','Author',2000)
            self.adder.execute('b','Known','Author',2000)
        network.assert_called_once()
        self.assertEqual(self.refs[0].provenance_kind,'verified_title')

    def test_changing_title_cannot_reuse_old_receipt(self):
        self.gate.record('Old','search')
        context=PaperContext(reference_library=[Paper('a','Old','A',2000)])
        update=UpdateReferenceTool()
        update.context=context.view({'reference_library'},{'reference_library'})
        update.provenance=self.gate; update.verifier=self.verify
        with patch.object(self.verify._searcher,'_search_auto',return_value='{"ok":true,"results":[]}'):
            result=update.execute('a',{'title':'Invented'})
        self.assertTrue(result.startswith('Error:'))
        self.assertEqual(context.reference_library[0].title,'Old')
        self.assertTrue(update.execute('a',{'provenance_kind':'pdf'}).startswith('Error:'))

    def test_missing_gate_fails_closed(self):
        tool=AddReferenceTool(); tool.set_add_func(self.refs.append)
        self.assertTrue(tool.execute('a','Known','A',2000).startswith('Error:'))
        self.assertEqual(self.refs,[])


if __name__=='__main__': unittest.main()
