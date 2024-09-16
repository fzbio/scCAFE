import numpy as np
from fanc.architecture.helpers import vector_enrichment_profile


def compartment_enrichment_profile(hic, hmm_proba, percentiles=(20.0, 40.0, 60.0, 80.0, 100.0),
                       symmetric_at=None,
                       exclude_chromosomes=(),
                       intra_chromosomal=True, inter_chromosomal=False,
                       collapse_identical_breakpoints=False,
                       ):



    mappable = hic.mappable()
    return vector_enrichment_profile(hic, hmm_proba, mappable=mappable,
                                     per_chromosome=True,
                                     percentiles=percentiles, symmetric_at=symmetric_at,
                                     exclude_chromosomes=exclude_chromosomes,
                                     intra_chromosomal=intra_chromosomal,
                                     inter_chromosomal=inter_chromosomal,
                                     collapse_identical_breakpoints=collapse_identical_breakpoints)