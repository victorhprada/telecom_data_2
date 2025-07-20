import unittest
import pandas as pd
import numpy as np
from src.preprocessing.prepare_data import *
from src.preprocessing.balance_data import *

class TestPreprocessing(unittest.TestCase):
    """Testes para funções de preprocessamento."""
    
    @classmethod
    def setUpClass(cls):
        """Setup inicial para todos os testes."""
        # Criar dados de exemplo
        cls.sample_data = pd.DataFrame({
            'customerID': ['001', '002', '003'],
            'Churn': [True, False, True],
            'tenure': [12, 24, 36],
            'Charges.Monthly': [50.0, 75.0, 100.0],
            'Contract': ['Month-to-month', 'One year', 'Two year']
        })
    
    def test_data_types(self):
        """Testar se os tipos de dados estão corretos."""
        self.assertTrue(self.sample_data['Churn'].dtype == bool)
        self.assertTrue(self.sample_data['tenure'].dtype == int)
        self.assertTrue(self.sample_data['Charges.Monthly'].dtype == float)
        
    def test_missing_values(self):
        """Testar se não há valores faltantes."""
        self.assertFalse(self.sample_data.isnull().any().any())
        
    def test_value_ranges(self):
        """Testar se os valores estão dentro dos ranges esperados."""
        self.assertTrue(all(self.sample_data['tenure'] >= 0))
        self.assertTrue(all(self.sample_data['Charges.Monthly'] > 0))

if __name__ == '__main__':
    unittest.main() 