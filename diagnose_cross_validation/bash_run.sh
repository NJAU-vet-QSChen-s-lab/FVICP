cd E:\R\diagnose_torch_models\pytorch\grid_search_b1
conda activate torch2.1.2_M40_cu12.1
python E:\R\diagnose_torch_models\pytorch\train_b1.py

cd E:\R\diagnose_torch_models\pytorch\grid_search_b2
conda activate torch2.1.2_M40_cu12.1
python E:\R\diagnose_torch_models\pytorch\train_b2.py

cd E:\R\diagnose_torch_models\pytorch
mkdir visualization
cd visualization
conda activate torch2.1.2_M40_cu12.1
python E:\R\diagnose_torch_models\pytorch\best_records_outputs.py

cd E:\R\diagnose_torch_models\pytorch\visualization
conda activate torch2.1.2_M40_cu12.1
python E:\R\diagnose_torch_models\pytorch\b1_visualization_train_validation_accuracy_merge_v3.py
python E:\R\diagnose_torch_models\pytorch\b2_visualization_train_validation_accuracy_merge_v3.py

cd E:\R\diagnose_torch_models\pytorch\visualization
conda activate torch2.1.2_M40_cu12.1
python E:\R\diagnose_torch_models\pytorch\shap_b1.py

cd E:\R\diagnose_torch_models\pytorch\visualization
conda activate torch2.1.2_M40_cu12.1
python E:\R\diagnose_torch_models\pytorch\visualization\b1_fascia_cross.py
#Accuracy: 96.44%
cd E:\R\diagnose_torch_models\pytorch\visualization
conda activate torch2.1.2_M40_cu12.1
python E:\R\diagnose_torch_models\pytorch\visualization\b2_fascia_cross.py
#Accuracy: 98.47%
