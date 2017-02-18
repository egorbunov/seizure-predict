from seiz.data_read import MatPatientDataReader
from tqdm import tqdm

for i in [1, 2, 3]:
    dr = MatPatientDataReader(i, "../data")
    td, n = dr.train_samples()
    possible_dims = set()
    with tqdm(range(n)) as timer:
        for t in td:
            possible_dims.add((t['sampling_rate'], t['signals'].shape[0]))
            timer.update()
    print(possible_dims)
