# Grounded language learning and GNNs

Tristan Karch, Laetitia Teodorescu

## Important to Discuss

1. In get_graph do we really need to extract ei and e as we are doing edge calculations with a transformer and « every body needs to pay attention to everybody »?
2. How are the wall treated in the function gnns.utils.get_entities ?
3. I think that the memory init should not be part of the SlotMemSparse model, this way we can get rid of the batch size argument and make it more flexible (related to Issue point 1)
4. 
	
## Todo
1. [OK] Understand memory mask —> memory is set to zero when episode is done to continue calculation on batches that are not done ?
3. [OK] Put preprocess obs dans format sinon ca va faire de la merde entre collect et act
4. Faire un assert pour que le preprocessGNN obs ne soit appliqué que si on est dans un GNNmodel. --> New separate file train\_rl_gnn.py 
5. Look at how memories are initialised in policy RNN from Bahdanau
	- Set to zero when ACmodel is initialized during construction (see base.py) 
	- Do they reset it to zero at the beginning of a new episode. Apparently no (or I can't find where)
6. Change base.py to accept matrices of memory instead of one memory vector
	- c'est un peu relou a faire, je me demande si on ne devrait pas s'affranchir un peu plus du code de Bahdanau et créer une sorte de base_gnn.py --> c'est l'option que j'ai choisi au final

## Issue
1. SlotMemSparse requires to fix the number of batches during construction but we want it to either process a single data (when it acts) or to process num\_frame\_per\_proc/reccurence (when it learns)
2. [OK] Something is not working in the implementation of the SparseTransformer (line 186:  qs, ks, vs = q[src], k[dest], v[dest])
--> Fixed it by putting ei = ei.to(device).type(torch.LongTensor) line 236 of gnns.utils
      