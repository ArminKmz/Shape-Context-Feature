import math, cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import tps

class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def cart2logpolar(self):
        '''
            return (rho, theta)
        '''
        rho = math.sqrt(self.x * self.x + self.y * self.y)
        theta = math.atan2(self.x, self.y)
        return (math.log(rho), theta)

    def dist2(self, other):
        return (self.x - other.x)**2 + (self.y - other.y)**2

class Shape:
    def __init__(self, shape=None, img=None):
        '''
            shape -> 2d list of [[x1, y1], ...],
                     default shape is canny edges
            self.shape -> shape
            self.shape_pts -> list of Point instead of lists.
            self.shape_contexts -> list of [arrays -> shape_context]
        '''
        self.img = img
        if shape is None:
            shape = utils.canny_edge_shape(img)
        self.shape = shape
        self.shape_pts = []
        for point in shape:
            self.shape_pts.append(Point(point[0], point[1]))
        self.shape_contexts = self.get_shape_contexts()

    def get_shape_contexts(self, angular_bins=12, radious_bins=None):
        '''
            angular_bins -> number of bins for angle.
            radious_bins -> number of bins for radious,
                            default is maximum radious.
            return -> list of shape context in image (bin array)
        '''
        # get maximum number of radious_bins
        if radious_bins is None:
            max_dist2 = 0
            for i in range(len(self.shape_pts)):
                for j in range(len(self.shape_pts)):
                    max_dist2 = max(max_dist2, self.shape_pts[i].dist2(self.shape_pts[j]))
            radious_bins = int(math.log(math.sqrt(max_dist2))) + 1
        shape_contexts = [np.zeros((radious_bins, angular_bins), dtype=float) for _ in range(len(self.shape_pts))]
        # compute bins
        for i in range(len(self.shape_pts)):
            for j in range(len(self.shape_pts)):
                if i == j:
                    continue
                pt = Point(self.shape_pts[j].x - self.shape_pts[i].x,
                           self.shape_pts[j].y - self.shape_pts[i].y)
                r, theta = pt.cart2logpolar()
                if r < 0:
                    x = 0
                else:
                    x = int(r)
                if theta == math.pi:
                    y = angular_bins - 1
                else:
                    y = int(angular_bins * ((theta + math.pi) / (math.pi + math.pi)))
                shape_contexts[i][x][y] += 1
        return [shape_context.reshape((radious_bins * angular_bins)) for shape_context in shape_contexts]

    def get_cost_matrix(self, Q, beta=.1, robust=False, dummy_cost=1):
        '''
            Q -> instance of Shape
            beta -> coefficient of tangent_angle_dissimilarity,
                    1-beta is coefficient of shape_context_cost
            return -> (cost matrix for matching a points
                      from shape_context1 to shape_context2,
                      flag -> dummies added or not
                              cif not -> 0
                              if added to P -> -n
                              if added to Q -> m)
        '''
        def normalize_histogram(hist, total):

            for i in range(hist.shape[0]):
                hist[i] /= float(total)
            return hist

        def shape_context_cost(nh1, nh2):
            '''
                nh1, nh2 -> normalized histogram
                return cost of shape context of
                two given shape context of the shape.
            '''
            cost = 0
            if nh1.shape[0] > nh2.shape[0]:
                nh1, nh2 = nh2, nh1
            nh1 = np.hstack([nh1, np.zeros(nh2.shape[0] - nh1.shape[0])])
            for k in range(nh1.shape[0]):
                if nh1[k] + nh2[k] == 0:
                    continue
                cost += (nh1[k] - nh2[k])**2 / (nh1[k] + nh2[k])
            return cost / 2.0

        def tangent_angle_dissimilarity(p1, p2):
            '''
                p1 -> Point 1
                p2 -> Point 2
                return -> tangent angle dissimilarity of
                          given two points
            '''
            theta1 = math.atan2(p1.x, p1.y)
            theta2 = math.atan2(p2.x, p2.y)
            return .5 * (1 - math.cos(theta1 - theta2))

        if robust:
            raise ValueError('robust=True not supported yet.')
        n, m = len(self.shape_pts), len(Q.shape_pts)
        flag = min(n, m) if (n != m) else 0
        if flag and (n < m):
            flag = -flag
        mx = max(n, m)
        C = np.zeros((mx, mx))
        for i in range(mx):
            if n <= i:
                for j in range(mx):
                    C[i, j] = dummy_cost
            else:
                p = self.shape_pts[i]
                hist_p = normalize_histogram(self.shape_contexts[i], n-1)
                for j in range(mx):
                    if m <= j:
                        C[i, j] = dummy_cost
                    else:
                        q = Q.shape_pts[j]
                        hist_q = normalize_histogram(Q.shape_contexts[j], m-1)
                        C[i, j] = (1-beta) * shape_context_cost(hist_p, hist_q)\
                            + beta * tangent_angle_dissimilarity(p, q)
        return C, flag

    def matching(self, Q):
        '''
            return -> two 2 x min(n, m) array.
                      (Pshape, Qshape) point i
                      from Pshape matched to
                      point i from Qshape.
        '''
        cost_matrix, flag = self.get_cost_matrix(Q)
        perm = linear_sum_assignment(cost_matrix)[1]
        Pshape = np.array(self.shape)
        Qshape = np.array(Q.shape)
        # removing dummy matched.
        if flag < 0:
            mn = -flag
            new_perm = perm[:mn]
            Qshape = Qshape[new_perm]
        elif flag > 0:
            mn = flag
            mask = perm < mn
            new_perm = perm[mask]
            Pshape = Pshape[mask]
            Qshape = Qshape[new_perm]
        return Pshape, Qshape

    def estimate_transformation(source, target):
        '''
            source -> n x 2 array of source points.
            target -> n x 2 array of source points.
            return -> bending energy, TPS class for transformation
        '''
        T = tps.TPS()
        BE = T.fit(source, target)
        return (BE, T)

    def shape_context_distance(self, Q_transformed, T):
        '''
            Q_transformed -> transformed target shape.
            T -> transformation function (TPS class)
            return -> shape context distance
        '''
        n, m = len(self.shape), len(Q_transformed.shape)
        cost_matrix = self.get_cost_matrix(Q_transformed)[0]
        ret1, ret2 = 0.0, 0.0
        for i in range(n):
            mn = 1e20
            for j in range(m):
                mn = min(mn, cost_matrix[i, j])
            ret1 += mn
        for j in range(m):
            mn = 1e20
            for i in range(n):
                mn = min(mn, cost_matrix[i, j])
            ret2 += mn
        return ret1 / n + ret2 / m

    def appearance_cost(source, target_transformed, img_p, img_q, std=1, window_size=3):
        '''
            source -> n x 2 array [source shape].
            target_transformed -> n x 2 array transformed target shape.
                                  [point i matched with point i from source]
            img_p -> source image.
            img_q -> target image.
            std -> scalar [standard deviation for guassian window].
            window_size -> size of guassian window.
            return -> appearance cost.
        '''
        def guassian_window(std, window_size):
            '''
                std -> scalar [standard deviation].
                window_size -> size of guassian window.
                return -> guassian window.
            '''
            window = np.zeros((window_size, window_size))
            for x in range(-(window_size//2), window_size//2 + 1):
                for y in range(-(window_size//2), window_size//2 + 1):
                    window[x][y] = math.exp(-(x*x + y*y) / (2 * std * std)) / (2 * math.pi * std * std)
            return window
        ret = 0
        G = guassian_window(std, window_size)
        for i in range(source.shape[0]):
            for x in range(-(window_size//2), window_size//2 + 1):
                for y in range(-(window_size//2), window_size//2 + 1):
                    px = min(int(x + source[i, 0]), img_p.shape[0]-1)
                    py = min(int(y + source[i, 1]), img_p.shape[1]-1)
                    Ip = int(img_p[px, py])
                    qx = min(int(x + target_transformed[i, 0]), img_q.shape[0]-1)
                    qy = min(int(y + target_transformed[i, 1]), img_q.shape[1]-1)
                    Iq = int(img_q[qx, qy])
                    ret += G[x + window_size//2, y + window_size//2] * (Ip - Iq)**2
        return ret / source.shape[0]

    def _distance(self, Q, w1, w2, w3, iterations=3):
        '''
            Q -> instance of Shape.
            w1 -> weight of Appearance Cost.
            w2 -> weight of Shape Contex distance.
            w3 -> weigth of Transformation Cost.
            iteration -> number of re-estimation of Transformation
                         estimation.
            return -> distance between two shapes.
        '''
        def transform_shape(Q, T):
            '''
                Q -> instance of Shape.
                T -> instance of TPS.
                return -> new Q which transformed with T.
            '''
            transformed_shape = []
            for q in Q.shape:
                Tq = T.transform(np.array(q).reshape((1, 2)))
                transformed_shape.append([Tq[0, 0], Tq[0, 1]])
            Q_transformed = Shape(transformed_shape, Q.img)
            return Q_transformed

        def transform_points(target_points, T):
            '''
                target_points -> n x 2 array of (x, y).
                T -> instance of TPS.
                return -> transform target_points with T.
            '''
            transformed_target = np.zeros_like(target_points)
            for i in range(target_points.shape[0]):
                new_pt = T.transform(target_points[i, :].reshape((1, 2)))
                transformed_target[i, :] = new_pt
            return transformed_target

        for i in range(iterations):
            source, target = self.matching(Q)
            BE, T = Shape.estimate_transformation(source, target)
            self = transform_shape(self, T)
        Q_transformed = transform_shape(Q, T)
        target_transformed = transform_points(target, T)
        AC = Shape.appearance_cost(source, target_transformed, self.img, Q.img)
        SC = self.shape_context_distance(Q, T)
        return w1 * AC + w2 * SC + w3 * BE

def distance(source_img, target_img, w1=1.6, w2=1, w3=.3):
    P = Shape(img=source_img)
    Q = Shape(img=target_img)
    return P._distance(Q, w1, w2, w3)

class utils:
    def canny_edge_shape(img, max_samples=100, t1=100, t2=200):
        '''
            return -> list of sampled Points from edges
                      founded by canny edge detector.
        '''
        edges = cv2.Canny(img, t1, t2)
        x, y = np.where(edges != 0)
        if x.shape[0] > max_samples:
            idx = np.random.choice(x.shape[0], max_samples, replace=False)
            x, y = x[idx], y[idx]
        shape = []
        for i in range(x.shape[0]):
            shape.append([x[i], y[i]])
        return shape
