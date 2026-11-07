## References

1. Dawkins R. 1976 *The selfish gene*. Oxford, UK: Oxford University Press.
2. Aunger R (ed.). 2000 *Darwinizing culture: the status of memetics as a science*. Oxford, UK: Oxford University Press.
3. Edmonds B. 2005 The revealed poverty of the gene-meme analogy: why memetics per se has failed to produce substantive results. *J. Memetics* **9**.
4. Mesoudi A, Whiten A, Laland KN. 2006 Towards a unified science of cultural evolution. *Behav. Brain Sci.* **29**, 329–383.
5. Tehrani JJ. 2013 The phylogeny of Little Red Riding Hood. *PLoS ONE* **8**, e78871.
6. da Silva SG, Tehrani JJ. 2016 Comparative phylogenetic analyses uncover the ancient roots of Indo-European folktales. *R. Soc. Open Sci.* **3**, 150645. [doi:10.1098/rsos.150645](https://doi.org/10.1098/rsos.150645)
7. Goh G, Cammarata N, Voss C, Carter S, Petrov M, Schubert L, Radford A, Olah C. 2021 Multimodal neurons in artificial neural networks. *Distill* **6**, e30.
8. Gurnee W, Horsley T, Guo ZC, Kheirkhah TR, Sun Q, Hathaway W, Nanda N, Bertsimas D. 2024 Universal neurons in GPT2 language models. *Trans. Mach. Learn. Res.*
9. Huh M, Cheung B, Wang T, Isola P. 2024 The platonic representation hypothesis. In *Proc. 41st Int. Conf. Machine Learning*.
10. Cammarata N. Neural networks are already clean. Goodfire. [already-clean.goodfire.com](https://already-clean.goodfire.com/) (accessed 7 November 2026).
11. Qwen Team. 2025 Qwen3 technical report. arXiv:2505.09388.
12. Grattafiori A *et al.* 2024 The Llama 3 herd of models. arXiv:2407.21783.
13. Saitou N, Nei M. 1987 The neighbor-joining method: a new method for reconstructing phylogenetic trees. *Mol. Biol. Evol.* **4**, 406–425.
14. Mantel N. 1967 The detection of disease clustering and a generalized regression approach. *Cancer Res.* **27**, 209–220.
15. Cohen J. 1960 A coefficient of agreement for nominal scales. *Educ. Psychol. Meas.* **20**, 37–46.
16. Marti L, Wu S, Piantadosi ST, Kidd C. 2023 Latent diversity in human concepts. *Open Mind* **7**, 79–92.
17. Heylighen F, Chielens K. 2009 Evolution of culture, memetics. In *Encyclopedia of complexity and systems science* (ed. RA Meyers). New York, NY: Springer.
18. Quian Quiroga R, Reddy L, Kreiman G, Koch C, Fried I. 2005 Invariant visual representation by single neurons in the human brain. *Nature* **435**, 1102–1107.
19. Chavalarias D, Cointet J-P. 2013 Phylomemetic patterns in science evolution: the rise and fall of scientific fields. *PLoS ONE* **8**, e54847. [doi:10.1371/journal.pone.0054847](https://doi.org/10.1371/journal.pone.0054847)
20. Kuhn T, Perc M, Helbing D. 2014 Inheritance patterns in citation networks reveal scientific memes. *Phys. Rev. X* **4**, 041036. [doi:10.1103/PhysRevX.4.041036](https://doi.org/10.1103/PhysRevX.4.041036)
21. Barbrook AC, Howe CJ, Blake N, Robinson P. 1998 The phylogeny of The Canterbury Tales. *Nature* **394**, 839. [doi:10.1038/29667](https://doi.org/10.1038/29667)
22. Chaturvedi S, Srivastava S, Roth D. 2018 Where have I heard this story before? Identifying narrative similarity in movie remakes. In *Proc. 2018 Conf. North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, vol. 2, pp. 673–678. [doi:10.18653/v1/N18-2106](https://doi.org/10.18653/v1/N18-2106)
23. Hatzel HO, Biemann C. 2024 Tell me again! A large-scale dataset of multiple summaries for the same story. In *Proc. 2024 Joint Int. Conf. Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)*, pp. 15732–15741. [aclanthology.org/2024.lrec-main.1366](https://aclanthology.org/2024.lrec-main.1366)
24. Choi D, Huang V, Meng K, Johnson DD, Steinhardt J, Schwettmann S. 2024 Scaling automatic neuron description. Transluce. [transluce.org/neuron-descriptions](https://transluce.org/neuron-descriptions) (accessed 20 October 2026).
25. Movva R, Peng K, Garg N, Kleinberg J, Pierson E. 2025 Sparse autoencoders for hypothesis generation. In *Proc. 42nd Int. Conf. Machine Learning*. arXiv:2502.04382.
26. Coogan P. 2006 *Superhero: the secret origin of a genre*. Austin, TX: MonkeyBrain Books.
27. Jha R, Zhang C, Shmatikov V, Morris JX. 2025 Harnessing the universal geometry of embeddings. arXiv:2505.12540. [doi:10.48550/arXiv.2505.12540](https://doi.org/10.48550/arXiv.2505.12540)

## Data, code and AI use

**Data and code.** Everything is in the study folder, `study/`, of [github.com/hammer-mt/phylomemetics](https://github.com/hammer-mt/phylomemetics). The code for this version is in `study/v2/`, its results in `study/results/v2/`, and the analysis plan in `study/v2/PLAN.md`. Version 1's code and results are unchanged beside them, and version 1 of the paper is in `study/paper/`. The tables in this paper are written from the committed results by `study/v2/analysis/paper_tables.py`, and the figures by the scripts in `study/v2/figs/`. Every explanation, every test text and every judge's verdict is committed, with the prompts the agents were given. The reference corpus and supplement are built from pinned public datasets by `study/v2/build_reference.py` and `study/v2/build_supplement.py`, and every document is listed in the results. Raw activations are not stored; the passes can be re-run with the scripts on a rented GPU. The code is under the MIT licence. Plot summaries and texts derived from them come from English Wikipedia and remain under CC BY-SA 4.0.

**Pre-registration.** The three experiments were specified, with falsifying results, before version 1 ran them. For this version the analysis plan was committed and pushed to the repository (commit `1f32fd1`, 02:55 UTC on 7 November 2026) before any of its models read a study text, so the repository's history records the order. The plan was reviewed beforehand by independent AI reviewers asked to find what was wrong with it. Section 18 of the plan logs the changes made after the freeze, each with its reason and with what was known at the time. The planted-story experiments that version 1 promised are pre-registered separately in `preregistration/`. None has run, and this version does not replace them.

**Authors' contributions.** Mike Taylor conceived the study and its theoretical framing, chose the Batman case and directed the work. For this version he asked for the method to be aligned with Cammarata's essay and for the weaknesses of version 1 to be fixed, and provided the GPU budget. Claude wrote the plan and the code for this version, ran the models and the analyses, and drafted the paper.

**AI use.** Claude is an AI model made by Anthropic, and is credited as an author for the contributions above. AI agents of the same model family also played every role in the explanation tests (explainer, tester, keyword analyst, kind judge, coverage judge and steering judge), each seeing only its own prompt, and carried out the adversarial reviews of the plan and the fact-check of this paper. Qwen3-32B repeated two of the judgments as a second model family. No human has audited the explanations or the judges' verdicts. The items and answer keys are committed so that an audit can be done.

**Competing interests.** None declared.
