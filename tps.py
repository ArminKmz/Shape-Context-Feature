import numpy as np
from scipy.spatial.distance import cdist


class TPS:
    def fit(self, source, target, reg=1e-3):
        '''
            source -> n x 2 array of source points.
            target -> n x 2 array of target points.
            reg -> regularization coefficient.
            return -> bending energy
        '''
        n = source.shape[0]
        K = TPS.U(cdist(source, source, 'sqeuclidean'))
        # regularaizing
        K += reg * np.eye(K.shape[0])
        P = np.concatenate((np.ones((n, 1)), source), axis=1)
        L = np.vstack([np.hstack([K, P]),
                       np.hstack([P.T, np.zeros((3, 3))])])
        Y = np.concatenate((target, np.zeros((3, 2))), axis=0)
        L_inv = np.linalg.inv(L)
        L_inv_Y = np.matmul(L_inv, Y)
        self.W = L_inv_Y
        self.source = source
        self.bending_energy = np.trace(np.matmul(np.matmul(self.W[:n, :].T, K), self.W[:n, :]))
        return self.bending_energy

    def transform(self, point):
        '''
            points -> 1 x 2 array representing (x, y).
            return -> 1 x 2 array of transformed point.
        '''
        dist = TPS.U(cdist(point, self.source, 'sqeuclidean'))
        tmp = np.array([[1, point[0, 0], point[0, 1]]])
        new_point = np.matmul(np.hstack([dist, tmp]), self.W)
        return new_point

    def U(dist):
        '''
            dist -> matrix of squared euclidean distance
                    between pairs.
            return -> [d * log(d)] elementwise on dist.
        '''
        ret = np.zeros_like(dist)
        mask = dist > 0
        ret[mask] = dist[mask] * np.log(dist[mask])
        return ret
