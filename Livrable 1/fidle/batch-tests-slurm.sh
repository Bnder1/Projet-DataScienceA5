#!/bin/bash
# -----------------------------------------------
#         _           _       _
#        | |__   __ _| |_ ___| |__
#        | '_ \ / _` | __/ __| '_ \
#        | |_) | (_| | || (__| | | |
#        |_.__/ \__,_|\__\___|_| |_|
#                              Fidle at IDRIS
# -----------------------------------------------
# Full_gpu ci - pjluc 2021
#
# Soumission :  sbatch  /(...)/fidle/VAE/batch_slurm.sh
# Suivi      :  squeue -u $USER

# ==== Job parameters ==============================================

#SBATCH --job-name="Fidle-ci"                          # nom du job
#SBATCH --ntasks=1                                     # nombre de tâche (un unique processus ici)
#SBATCH --gres=gpu:1                                   # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --cpus-per-task=10                             # nombre de coeurs à réserver (un quart du noeud)
#SBATCH --hint=nomultithread                           # on réserve des coeurs physiques et non logiques
#SBATCH --time=05:00:00                                # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output="CI_%j.out"                           # nom du fichier de sortie
#SBATCH --error="CI_%j.err"                            # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --mail-user=Jean-Luc.Parouty@grenoble-inp.fr
#SBATCH --mail-type=ALL

# ==== Notebook parameters =========================================

MODULE_ENV="tensorflow-gpu/py3/2.4.0"
NOTEBOOK_DIR="$WORK/fidle/fidle"

FIDLE_OVERRIDE_PROFILE="./ci/full_gpu.yml"

NOTEBOOK_SRC1="02-running-ci-tests.ipynb"
NOTEBOOK_SRC2="03-ci-report.ipynb"

# ==================================================================

export FIDLE_OVERRIDE_PROFILE

echo '------------------------------------------------------------'
echo "Start : $0"
echo '------------------------------------------------------------'
echo "Job id        : $SLURM_JOB_ID"
echo "Job name      : $SLURM_JOB_NAME"
echo "Job node list : $SLURM_JOB_NODELIST"
echo '------------------------------------------------------------'
echo "Notebook dir  : $NOTEBOOK_DIR"
echo "Notebook src1 : $NOTEBOOK_SRC1"
echo "Notebook src2 : $NOTEBOOK_SRC2"
echo "Environment   : $MODULE_ENV"
echo '------------------------------------------------------------'
env | grep FIDLE_OVERRIDE | awk 'BEGIN { FS = "=" } ; { printf("%-35s : %s\n",$1,$2) }'
echo '------------------------------------------------------------'

# ---- Module

module purge
module load "$MODULE_ENV"

# ---- Run it...

cd $NOTEBOOK_DIR

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --output "${NOTEBOOK_SRC1%.*}==${SLURM_JOB_ID}==.ipynb" --execute "$NOTEBOOK_SRC1"
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --output "${NOTEBOOK_SRC2%.*}==${SLURM_JOB_ID}==.ipynb" --execute "$NOTEBOOK_SRC2"

echo 'Done.'
