# Development notes
--------------------

## August 26:
-------------


- Added MoE file, and started routing training to MoE object.
- MoE object takes the model that is already set up and replicates to N experts
- Right now need to work on dispatch function to dispatch slices of the input to each expert
    - Right now input seems to be 7 feature tensors with shape [395, N]. 
        - gt_s=None, gt_o=None, gt_p_vec=None are ground truth labels.

- Placed debugger right before fwd call. Need to fix displace function to get it to properly handle all 7 tensors. 
- Should investigate the optimizers too. Right now we have them working for all experts, we may need them for each activated expert.
- Why is the input dim [395, N]? Shouldn't it be 12, N? Investigate this too.
    - Seems 395 is related to batch size directly. When increased batch size to 64, we get [2243,N]


## August 27:
-------------
- Figured out how to dispatch the slices of inputs to each expert.
    - Expert dims are first concatinated on axis=1, so that the input ot gating layer is [batch_size, <s,o,p features>]
    - Once gates are  calculated, experts then execute fwd call.
    - Each expert contains an S biased score, O biased score, and P biased score (what is biased here? Not sure yett...)
    - Need to combine all these scores together into one, then return to main learner flow. (MoE -> combine()
        - Issue with combination. self.w_gate is size [1580,1] but sttiched score is size [395, N]
        - Combination successful.

    - [info] best training loss 0.1101 after 46 training epoch (Original)
    
        - [info] best training loss 0.1025 after 50 training epoch (MoE, 10 experts, k = 4)
        - [info] best training loss 0.0991 after 48 training epoch (MoE, 100 experts, k = 10)

    - Independent Classifier
        - Results from original code:
            - [info] best training loss 0.1101 after 46 training epoch
                [setting] overall
                    detection mean AP:	0.2746
                    detection recall@50:	0.1839
                    detection recall@100:	0.2097
                    tagging precision@1:	0.6350
                    tagging precision@5:	0.4950
                    tagging precision@10:	0.3760
                [setting] zero-shot
                    detection mean AP:	0.0085
                    detection recall@50:	0.0185
                    detection recall@100:	0.0185
                    tagging precision@1:	0.1918
                    tagging precision@5:	0.0986
                    tagging precision@10:	0.0575
                [setting] generalized zero-shot
                    detection mean AP:	0.0039
                    detection recall@50:	0.0185
                    detection recall@100:	0.0185
                    tagging precision@1:	0.0411
                    tagging precision@5:	0.0192
                    tagging precision@10:	0.0178
    
        - Results from MoE code (100 experts, k=10):
            - [info] best training loss 0.1104 after 45 training epoch
                [setting] overall
                        detection mean AP:      0.2544
                        detection recall@50:    0.1626
                        detection recall@100:   0.1868
                        tagging precision@1:    0.6450
                        tagging precision@5:    0.4810
                        tagging precision@10:   0.3555
                [setting] zero-shot
                        detection mean AP:      0.0243
                        detection recall@50:    0.0255
                        detection recall@100:   0.0255
                        tagging precision@1:    0.2329
                        tagging precision@5:    0.0986
                        tagging precision@10:   0.0548
                [setting] generalized zero-shot
                        detection mean AP:      0.0037
                        detection recall@50:    0.0139
                        detection recall@100:   0.0208
                        tagging precision@1:    0.0411
                        tagging precision@5:    0.0247
        - Results from MoE code (256 experts, k = 4)
        - [info] best training loss 0.0951 after 48 training epoch
            [setting] overall
                detection mean AP:      0.2560
                detection recall@50:    0.1694
                detection recall@100:   0.1965
                tagging precision@1:    0.6200
                tagging precision@5:    0.4460
                tagging precision@10:   0.3480
            [setting] zero-shot
                detection mean AP:      0.0113
                detection recall@50:    0.0139
                detection recall@100:   0.0139
                tagging precision@1:    0.1644
                tagging precision@5:    0.0740
                tagging precision@10:   0.0438
            [setting] generalized zero-shot
                detection mean AP:      0.0023
                detection recall@50:    0.0093
                detection recall@100:   0.0139
                tagging precision@1:    0.0137
                tagging precision@5:    0.0110
                tagging precision@10:   0.0137
        - Results from MoE code (256 experts, k = 4) -- NO INIT WEIGHTS
        - [info] best training loss 0.1049 after 48 training epoch

        ## 100 epochs, batchs size 64
            [info] best training loss 0.0636 after 78 training epoch
            [setting] overall
                detection mean AP:	0.2691
                detection recall@50:	0.1690
                detection recall@100:	0.1986
                tagging precision@1:	0.6750
                tagging precision@5:	0.4990
                tagging precision@10:	0.3740
            [setting] zero-shot
                detection mean AP:	0.0091
                detection recall@50:	0.0162
                detection recall@100:	0.0162
                tagging precision@1:	0.2329
                tagging precision@5:	0.0932
                tagging precision@10:	0.0534
            [setting] generalized zero-shot
                detection mean AP:	0.0021
                detection recall@50:	0.0116
                detection recall@100:	0.0139
                tagging precision@1:	0.0137
                tagging precision@5:	0.0164
                tagging precision@10:	0.0205