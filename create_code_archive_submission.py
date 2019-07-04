import os, sys
import patoolib


out_file = "code+Readme.zip"
if os.path.exists(out_file):
    os.remove(out_file)

submission_files = ["data.py",
                    "layers.py",
                    "utils.py",
                    "tools.py",
                    # baselines
                    "generate_feats.py",
                    "baseline_patches.py",
                    "baseline_score_regressor.py",
                    "baseline_score_regressor_test.py",
                    # DCGAN
                    "DCGAN.py",
                    "train_DCGAN.py",
                    "DCGAN_Scorer.py",
                    "train_DCGAN_for_score.py",
                    "test_DCGAN_scorer.py",
                    # SRM
                    "StackedSRM.py",
                    "train_stackedSRM.py",
                    # MCGAN
                    "MCGAN.py",
                    "train_MCGAN.py",
                    # FullresGan
                    "FullresGAN.py",
                    "train_FullresGAN.py",
                    # Generation Pipeline
                    "test_GAN_SRM_Scorer.py",
                    # Experiments
                    "extract_features.py",
                    "run_experiments.py",
                    "baseline_score_generated.py",
                    "baseline_score_labeled.py",
                    # Others
                    "README.md",
                    "requirements.txt"
                   ]

patoolib.create_archive(out_file, submission_files)
    

