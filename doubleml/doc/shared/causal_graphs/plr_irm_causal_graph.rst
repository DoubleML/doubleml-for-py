.. graphviz::
   :align: center
   :caption: Causal diagram

   digraph {
        nodesep=1;
        ranksep=1;
        rankdir=LR;
        { node [shape=circle, style=filled]
          Y [fillcolor="#56B4E9"]
          D [fillcolor="#F0E442"]
          V [fillcolor="#F0E442"]
          X [fillcolor="#D55E00"]
        }
        Y -> D -> V [dir="back"];
        Y -> X [dir="back"];
   }