import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from autonlp.flags import Flags
from autonlp.features.cleaner import Preprocessing_NLP


class TestclassCleaner(object):
    """
    Test class Preprocessing_NLP

    Input :
        flags : Instance of Flags class object
        data (DataFrame) : data with a column flags.column_text and a column flags.target (optional)
    """

    flags_dict_info = {
        "column_text": "column_text",
        "target": "target",
        "outdir": "./logs_test",
        "apply_small_clean": True,
        "apply_spacy_preprocessing": False,
        "apply_entity_preprocessing": False,
    }
    flags = Flags().update(flags_dict_info)

    data = pd.DataFrame({flags.column_text: ["\ntext  "+str(i) for i in range(100)],
                         flags.target: ["target1" if i//2 else "target2" for i in range(100)]})

    # expected output :
    data_output = pd.DataFrame({flags.column_text: [" text " + str(i) for i in range(100)],
                                flags.target: ["target1" if i // 2 else "target2" for i in range(100)]})

    def test_cross_validation(self):

        pre = Preprocessing_NLP(self.data, self.flags)
        data, doc_spacy_data = pre.transform(self.data)

        assert_frame_equal(data, self.data_output)
        assert isinstance(data, pd.DataFrame)
        assert len(data) == len(self.data)
        assert doc_spacy_data is None
        assert '  ' not in data[self.flags.column_text].iloc[0]
        assert (data.index == self.data.index).all()