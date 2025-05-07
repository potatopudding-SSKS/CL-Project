Before running "pip install -r requirements.txt", make sure the Python version is 3.12 or lower

If creating a conda environment, the following command can be used:
    conda create -n "newenv" python=3.12.0 ipython

For training, the order of running scripts is:
1. comment_scraper.py
2. text_aggregator.py
3. corpus_annotator.py
4. splitter.py
5. crf.py

For inference, the folder "final-pipeline" is to be used