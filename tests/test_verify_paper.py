import json
import unittest
from unittest.mock import patch
from src.tools.builtin.search import VerifyPaperTool


class VerifyTests(unittest.TestCase):
    def verify(self, papers, title='Target paper', **kwargs):
        tool = VerifyPaperTool()
        with patch.object(tool._searcher, '_search_auto', return_value=json.dumps({'ok':True,'results':papers})):
            return json.loads(tool.execute(title, **kwargs))

    def test_wrong_first_result_is_not_verified(self):
        result=self.verify([{'title':'A fast sweeping method for Eikonal equations','authors':'Hongkai Zhao'}],
                           title='Solution of the eikonal equation by a finite-difference method', authors='Fuhao Qin')
        self.assertFalse(result['verified'])

    def test_matches_later_candidate_and_normalizes_punctuation(self):
        result=self.verify([{'title':'Wrong'}, {'title':'TARGET-paper','authors':'Alice Smith','year':2005,'doi':'10.1/test'}],
                           authors='Alice Smith',year=2005,doi='https://doi.org/10.1/TEST')
        self.assertTrue(result['verified'])

    def test_other_metadata_does_not_affect_title_verification(self):
        paper={'title':'Target paper','authors':'Alice Smith','year':2005,'doi':'10.1/a'}
        for constraint in [{'authors':'Bob Jones'},{'year':2004},{'doi':'10.1/b'}]:
            result = self.verify([paper],**constraint)
            self.assertTrue(result['verified'])
            self.assertEqual(result['checked_fields'], ['title'])

    def test_schema_requests_only_title(self):
        self.assertEqual(set(VerifyPaperTool().get_parameters()['properties']), {'title'})

    def test_ambiguous_versions_not_automatically_selected(self):
        result=self.verify([{'title':'Target paper','year':2004},{'title':'Target paper','year':2005}])
        self.assertTrue(result['verified'])
        self.assertEqual(result['match_count'],2)
        self.assertNotIn('year', result)

    def test_related_title_does_not_pass(self):
        self.assertFalse(self.verify([{'title':'Target paper extended'}])['verified'])

    def test_empty_title_does_not_search(self):
        tool=VerifyPaperTool()
        with patch.object(tool._searcher, '_search_auto') as search:
            self.assertFalse(json.loads(tool.execute('  -- '))['verified'])
        search.assert_not_called()

    def test_network_failure_is_distinct_from_no_results(self):
        tool=VerifyPaperTool()
        with patch.object(tool._searcher,'_search_auto',side_effect=TimeoutError('timeout')):
            result=json.loads(tool.execute('Target paper'))
        self.assertEqual(result['status'],'search_failed')
        self.assertEqual(self.verify([])['status'],'not_found')


if __name__=='__main__': unittest.main()
