..
   ReaL documentation master file, created by
   sphinx-quickstart on Mon Jun 10 10:57:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##################################
 Welcome to ReaL's documentation!
##################################

*****************
 🚀 Get Started 🚀
*****************

For users new to ReaL, we recommend starting with the :doc:`quickstart`
section to learn how to run simple experiments on a local node. If you
have multiple nodes available, please read the :doc:`distributed`
section to learn how to run experiments on a cluster. These tutorials
cover the basic usage of the implemented algorithms in ReaL, including
SFT, Reward Modeling, DPO, and PPO, and do not require understanding the
code structure.

For advanced users, we recommend proceeding to the :doc:`customization`
section to learn how to customize the algorithms and models in ReaL.
This requires an understanding of how an algorithm and its experiment
configuration are defined in ReaL (i.e., as a dataflow graph), but
understanding the system-wide implementation (e.g., model workers) is
not mandatory.

For potential developers, please refer to the :doc:`impl` and the
:doc:`arch` sections for a deeper understanding of the system
architecture.

Besides these illustrations, we present the reference manual of various
configuration objects in the :doc:`expconfig` section, and a brief
overview of the system architecture in the :doc:`intro` section.

**************
 ⭐ Contents ⭐
**************

.. toctree::
   :maxdepth: 3

   intro
   install
   expconfig
   quickstart
   distributed
   customization
   impl
   arch

   contributing
