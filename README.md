# ARCANE

<p align="center" width="100%">
  <img width="35%" src="https://github.com/Adversarian/arcane/blob/main/ARCANE.jpg" />
</p>

This repository contains my work for my Master's Thesis titled:
**ARCANE**: <ins>**A**</ins>dversarial <ins>**R**</ins>obustness using <ins>**C**</ins>lass-condition<ins>**A**</ins>l ge<ins>**N**</ins>erative mod<ins>**E**</ins>ls


ARCANE is a novel framework whose aim is to provide adversarial robustness to classifier model using class-conditional generative models. By projecting a given potentially adversarial sample onto the distribution learned through a generative model, ARCANE is able to purify a given sample to produce clean class labels. Moreover, a small classifier is able to detect whether a given sample contains adversarial noise with a set of 6 features that are extracted from the sample at inference time.