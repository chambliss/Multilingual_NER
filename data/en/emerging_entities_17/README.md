# Emerging Entities dataset

This is the dataset for the [WNUT 17 Emerging Entities task](https://noisy-text.github.io/2017/emerging-rare-entities.html), with supplemental information. 

The goal is to provide high-variance data, with very few repeated surface forms. This means, for example, you might see the location "London", but it won't be very frequent. Instead the documents here have a wide range of different entities, so learning that one particular one is an entity won't help much. 

If you'd like to read the full paper, it's here: [Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition](http://www.derczynski.com/sheffield/papers/emerging-wnut.pdf)

## Data description

The data is annotated for named entities: 

1. Person
1. Location
1. Group
1. Creative work
1. Corporation
1. Product

Text comes from a few sources. They are: YouTube comments, Stack Overflow responses, Twitter text around four separate major events in 2016/17, unfiltered Twitter text from 2010, and Reddit comments. The dataset is in English. In most cases, data has been filtered to prefer text that is likely to contain named entities.

## Data statement

Annotators were from Crowdflower, a crowdsourcing aggregation site, and asked to be from one of the following regions: USA, UK, New Zealand, Ireland, Jamaica, Australia, Canada, Botswana, South Africa. Adjudication was performed in four separate pieces, by annotators who were academics in NLP, working in Japan, USA, Ireland and the Netherlands. Two of the adjudicators were native speakers, from different regions; the other two were highly proficient L2 English users, having also lived in the UK for multiple years.

## Usage and acknowledgment

This data is distributed under the [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

Send pull requests to submit annotation amendments.

If you use this data, please recognize our hard work and cite the relevant paper:

Leon Derczynski, Eric Nichols, Marieke van Erp, Nut Limsopatham; 2017. **Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition**. In *Proceedings of the Workshop on Noisy, User-generated Text*, at EMNLP. [[pdf](http://noisy-text.github.io/2017/pdf/WNUT18.pdf)]


