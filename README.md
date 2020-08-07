# Grounded language learning and GNNs

Tristan Karch, Laetitia Teodorescu

## Important to Discuss

1. In get_graph do we really need to extract ei and e as we are doing edge calculations with a transformer and « every body needs to pay attention to everybody »?
2. How are the wall treated in the function gnns.utils.get_entities ?
3. I think that the memory init should not be part of the SlotMemSparse model, this way we can get rid of the batch size argument and make it more flexible (related to Issue point 1)
4. What format do we want to store the memories (Proc, Num_frame, Slots, Fmem) or (Proc_Num_frame, Slots, Fmem) ?
Smartest way is memory: (Slots\*Proc, Fmem) and memories (Num_frame, Slots\*Proc, Fmem) --> Ca permet de traiter intelligemment les environnemnets en paralleles avec les batchs.

	
## Todo

La liste dynamique des choses à faire. Je laisse certains points qui ont étés traités mais pour lesquels j'ai encore quelques interrogations


1. [OK] Understand memory mask —> memory is set to zero when episode is done to continue calculation on batches that are not done ?
4. Faire un assert pour que le preprocessGNN obs ne soit appliqué que si on est dans un GNNmodel. --> New separate file train\_rl_gnn.py 
5. Look at how memories are initialised in policy RNN from Bahdanau
	- Set to zero when ACmodel is initialized during construction (see base.py) 
	- Do they reset it to zero at the beginning of a new episode. Apparently no (or I can't find where)
1. Bien intégrer que dans collect_experiences il y a autant de batch que d'environnements paralleles donc c'est pratique on connait deja le nombre de batch en amont. De la meme facon dans update parameters on peut connaitres le nombre de batch qui est (num_proc\*num_frames_per_procs) / recurrence


## Issue
1. SlotMemSparse requires to fix the number of batches during construction but we want it to either process a single data (when it acts) or to process num\_frame\_per\_proc/reccurence (when it learns)
2. [OK] Something is not working in the implementation of the SparseTransformer (line 186:  qs, ks, vs = q[src], k[dest], v[dest])
--> Fixed it by putting ei = ei.to(device).type(torch.LongTensor) line 236 of gnns.utils
3. Add to change torch.arange(nelems).type(torch.LongTensor)]) in line 6 of model_gnn (see with laetitia)
4. Problem all memory slots have same value (look with laetitia)


## Important design choices

- Use previous action in update of memory lstm (The agent needs to know its orientation)
- Init memory 
- __Specialization mechanism__ in memory update to make memories diverge from same init (attention stochastique)
- Language:
	- aggreagtion
	- update memory
	- relational model
	- do we use words or sentences as language atom. 

