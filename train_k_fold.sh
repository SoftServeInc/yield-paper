array=( 0 1 2 3 4 5 6 7 8 9 )

data_dir=/data
for i in "${array[@]}"
do
     python train.py --data_path    ${data_dir}/mapped_Buchwald_Hartwig/train_cv_${i}.csv  --features_path ${data_dir}/mapped_Buchwald_Hartwig/train_cv_${i}_feat_rdkit.csv --dataset_type regression  --separate_val_path ${data_dir}/mapped_Buchwald_Hartwig/test_cv_${i}.csv  --separate_test_path ${data_dir}/mapped_Buchwald_Hartwig/test_cv_${i}.csv  --separate_val_features_path ${data_dir}/mapped_Buchwald_Hartwig/test_cv_${i}_feat_rdkit.csv  --separate_test_features_path ${data_dir}/mapped_Buchwald_Hartwig/test_cv_${i}_feat_rdkit.csv --save_dir ./output/Buchwald_Hartwig_f${i}/ --reaction --epochs 110 --metric r2  --batch_size 64  --hidden_size 1000  --bias --depth 3 --depth_diff 1 --ffn_num_layers 3 --dropout 0.25 --activation LeakyReLU --no_features_scaling
     python predict.py --test_path  ${data_dir}/mapped_Buchwald_Hartwig/train_cv_${i}.csv   --features_path ${data_dir}/mapped_Buchwald_Hartwig/train_cv_${i}_feat_rdkit.csv --checkpoint_dir  ./output/Buchwald_Hartwig_f${i}/fold_0/model_0/ --preds_path  ./output/Buchwald_Hartwig_f${i}/train_preds.csv
     python predict.py --test_path  ${data_dir}/mapped_Buchwald_Hartwig/test_cv_${i}.csv   --features_path ${data_dir}/mapped_Buchwald_Hartwig/test_cv_${i}_feat_rdkit.csv --checkpoint_dir  ./output/Buchwald_Hartwig_f${i}/fold_0/model_0/ --preds_path  ./D-MPNN/output/Buchwald_Hartwig_f${i}/test_preds.csv --middle_representation_path ./output/Buchwald_Hartwig_f${i}/middle_layer_representations.csv --last_ffn_representation_path ./output/Buchwald_Hartwig_f${i}/last_layer_representations.csv

done

for i in "${array[@]}"
do
     python train.py --data_path    ${data_dir}/mapped_Suzuki_Miyaura/train_cv_${i}.csv  --features_path ${data_dir}/mapped_Suzuki_Miyaura/train_f${i}_feat_rdkit.csv --dataset_type regression  --separate_val_path ${data_dir}/mapped_Suzuki_Miyaura/test_cv_${i}.csv  --separate_test_path ${data_dir}/mapped_Suzuki_Miyaura/test_cv_${i}.csv  --separate_val_features_path ${data_dir}/mapped_Suzuki_Miyaura/test_f${i}_feat_rdkit_reactant.csv  --separate_test_features_path ${data_dir}/mapped_Suzuki_Miyaura/test_f${i}_feat_rdkit_reactant.csv --save_dir /home/dzvinka/D-MPNN/output/Suzuki_myiara_f${i}/ --reaction --epochs 65 --metric r2  --batch_size 256  --hidden_size 1000  --diff_hidden_size 2000 --bias --depth 1 --depth_diff 0 --ffn_num_layers 3 --dropout 0.1 --activation LeakyReLU --no_features_scaling
     python predict.py --test_path  ${data_dir}/mapped_Suzuki_Miyaura/train_cv_${i}.csv   --features_path ${data_dir}/mapped_Suzuki_Miyaura/train_cv_${i}_feat_rdkit.csv --checkpoint_dir  ./output/Suzuki_Miyaura_f${i}/fold_0/model_0/ --preds_path  ./output/Suzuki_Miyaura_f${i}/train_preds.csv
     python predict.py --test_path  ${data_dir}/mapped_Suzuki_Miyaura/test_cv_${i}.csv   --features_path ${data_dir}/mapped_Suzuki_Miyaura/test_cv_${i}_feat_rdkit.csv --checkpoint_dir  ./output/Suzuki_Miyaura_f${i}/fold_0/model_0/ --preds_path  ./output/Suzuki_Miyaura_f${i}/test_preds.csv --middle_representation_path ./output/Suzuki_Miyaura_f${i}/middle_layer_representations.csv --last_ffn_representation_path ./output/Suzuki_Miyaura_f${i}/last_layer_representations.csv
done


