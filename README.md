# The Effects of Patent Filing Acceleration on the Evolution of Technological Innovation

Author: [Ryan Steed](https://rbsteed.com)  
[DataMASTER Fellowship](https://math.columbian.gwu.edu/data-master) 2018-19  

Acknowledgement and thanks to:
- [Professor Rahul Simha](https://www.seas.gwu.edu/rahul-simha) for research mentorship and guidance
- [George Washington University Colonial One High Performance Computing](https://colonialone.gwu.edu/) for compute resources and data storage
- [PatentsView](http://www.patentsview.org) for easy public API access to USPTO patent data

## Abstract
The 2011 America Invents Act (AIA) disrupted the United States patenting process, replacing first-to-invent policy with the first-to-file system popular in Europe and sparking a furious debate over the impact of the policy on innovation and technological growth. The institution of patent protection, an integral part of intellectual property law and economic competition, is fundamentally changed by its passage, but many of its effects are unknown and unexplored. How does accelerating the process of patent filing affect the evolution of technological innovation?

To investigate these effects, I intend to 1) identify key properties of the patent citation network that are affected by the AIA, 2) employ a time series model to forecast network evolution for several specific patent classes, and 3) compare the rate of evolution before and after the implementation of the AIA.

The topological structure of the citation network alone may provide useful insight into the effects of the AIA on patent filing behavior. I hypothesize that many new prolific firms and inventors (as measured by H-indices) will become prominent in a denser citation network while already-prolific firms will not change output significantly, resulting in a more modular network. Beyond static topological analysis, I anticipate that the AIA has a catalyzing effect on network evolution, increasing rates of technology proliferation and mimicry in the citation network.

View the full project proposal [here](docs/RyanSteed_ProjectProposal.pdf).

## Index
|Contents | Description |
 --- | --- 
`app/` | The application source code.
`data/` | A folder for loaded data (graphs, patent trees, and custom queries).
`docs/` | API, project, and data exploration documentation.
`logs/` | Storage location for server logs, named by environment.
`scripts/` | Individual use scripts for slurm data collection and processing jobs and R analysis.
`slurm/` | Storage location for slurm log files.
`env.yml` | Dependencies for `conda` environment.
`main.py` | Driver script for the application containing API [endpoints](#api).

## API


---
© Ryan Steed 2019


