.. graphviz::
   :align: center
   :caption: Causal diagram

   digraph {
        nodesep=1;
        ranksep=1;
        rankdir=LR;
        { node [shape=circle, style=filled]
          Y [fillcolor="#56B4E9"]
          D [fillcolor="#56B4E9"]
          Z [fillcolor="#F0E442"]
          V [fillcolor="#F0E442"]
          X [fillcolor="#D55E00"]
        }

        Z -> V [dir="back"];
        D -> X [dir="back"];
        Y -> D [dir="both"];
        X -> Y;
        Z -> X [dir="back"];
        Z -> D;

        { rank=same; Y D }
        { rank=same; Z X }
	    { rank=same; V }
   }