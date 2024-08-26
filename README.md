<h1 align="center">ChatEvaluation using ChatGPT and Llama2.</h1>

<p align="center">
<a href="https://arxiv.org/abs/2308.06502"><b>Three Ways of Using Large Language Models to Evaluate Chat, on Arxiv.
</b></a><br>
<i>Accepted to <a href="https://dstc11.dstc.community/workshop/accepted-papers">DST11 workshop</a>, 2023, Prague</i></br>
<!-- <a href="Poster"><b>Poster</b></a><br>-->
</p>

<p>&nbsp;</p>

# Chateval Package

## Installation
```
# Optional for reinstallation
conda deactivate; rm -rf env; 
# Installing new conda environment and editable pip moosenet package
conda env create --prefix ./env -f environment.yml \
  && conda activate ./env \
  && pip install -e .[dev] 
```

<!-- link the prompts, says prompts with examples works best link the llama2 syntax-->

## Acknowledgements
This work was co-funded by Charles University projects GAUK 40222, SVV 260575 and the European Union (ERC, NG-NLG, 101039303).
<img src="https://ufal.mff.cuni.cz/~odusek/2024/images/LOGO_ERC-FLAG_FP.png" alt="erc-logo" height="150"/>
