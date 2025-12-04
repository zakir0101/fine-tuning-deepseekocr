# In a Kaggle notebook cell
echo "starting download .."
apt-get install git-lfs
git lfs install
git clone https://github.com/zakir0101/IGCSE_DATA.git 
cd IGCSE_DATA && git lfs pull
echo "[DONE] Download finish .."
