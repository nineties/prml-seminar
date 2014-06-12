# -*- coding: utf-8 -*-
from copy import copy
from itertools import product

#== 変数消去(+枝刈り)によるベイジアンネットワーク ==

class Factor:
    # xs:  ファクター内の変数リスト
    # sz:  各変数の取る値の個数のリスト
    # tbl: ファクターの値のテーブル
    def __init__(self, xs, sz, tbl):
        self.xs = xs
        self.sz = sz
        self.tbl = tbl
        # インデックス計算に使うオフセット
        self.stride = {}
        stride = 1
        for i in reversed(range(len(sz))):
            self.stride[xs[i]] = stride
            if i > 0:
                stride *= sz[i-1]
        self.title = "phi(" + ",".join(xs) + ")"

    # 変数の番号
    def index(self, x):
        return self.xs.index(x)

    # 変数の取る値の個数
    def xsize(self, x):
        return self.sz[self.xs.index(x)]

    # Φ(xs1=vs1, xs2 = vs2, ...) を返す.
    def lookup(self, xs, vs): 
        i = 0
        for j in range(len(xs)):
            x = xs[j]
            if not(x in self.xs): continue
            i += self.stride[x]*vs[j]
        return self.tbl[i]

    def __mul__(self, other):
        if other==1: return self
        xs = copy(self.xs)
        sz = copy(self.sz)
        for i in range(len(other.xs)):
            if other.xs[i] in xs: continue
            xs.append(other.xs[i])
            sz.append(other.sz[i])
        tbl = [self.lookup(xs, vs)*other.lookup(xs, vs)\
               for vs in product(*map(range, sz))]
        return Factor(xs, sz, tbl)

    # ファクターの比
    def __div__(self, other):
        if other==1: return self
        xs = copy(self.xs)
        sz = copy(self.sz)
        for i in range(len(other.xs)):
            if other.xs[i] in xs: continue
            xs.append(other.xs[i])
            sz.append(other.sz[i])
        tbl = [self.lookup(xs, vs)/other.lookup(xs, vs)\
               for vs in product(*map(range, sz))]
        return Factor(xs, sz, tbl)

    # 変数xについて周辺化
    def marginalize(self, x):
        xidx = self.index(x)
        xs = []
        sz = []
        for i in range(len(self.xs)):
            if i == xidx: continue
            xs.append(self.xs[i])
            sz.append(self.sz[i])
        tbl = []
        for vs in product(*map(range, sz)):
            s = 0
            for v in range(self.sz[xidx]):
                s += self.lookup(xs + [x], list(vs) + [v])
            tbl.append(s)
        return Factor(xs, sz, tbl)

    # 和が1になるように正規化
    def normalize(self):
        Z = sum(self.tbl)
        for i in range(len(self.tbl)):
            self.tbl[i] /= Z
        return self

    # x=vを満たさない欄を0にしたファクターを返す.
    def apply_evidence(self, x, v):
        if not(x in self.xs): return self
        tbl    = copy(self.tbl)
        stride = self.stride[x]
        sz     = self.sz[self.index(x)]
        for i in range(len(tbl)):
            if (i/stride)%sz != v: tbl[i] = 0.0
        return Factor(self.xs, self.sz, tbl)

    # 枝刈り
    def remove_evidence_node(self, x, v):
        if not(x in self.xs): return self
        new_tbl = []
        stride = self.stride[x]
        sz     = self.sz[self.index(x)]
        for i in range(len(self.tbl)):
            if (i/stride)%sz == v:
                new_tbl.append(self.tbl[i])
        new_xs = copy(self.xs)
        new_sz = copy(self.sz)
        new_xs.remove(x)
        new_sz.remove(sz)
        return Factor(new_xs, new_sz, new_tbl)

    def __repr__(self):
        prec = 10
        formatted = ""

        # 左のカラム幅
        lw = max(
            max(map(lambda x: len(str(self.xsize(x))), self.xs)),
            max(map(lambda x: len(x), self.xs))
            ) + 2

        # 一番右のカラムの幅
        rw = max(len(self.title), prec+2) + 2

        # ヘッダ
        formatted += "|".join(map(lambda x: str.center(x, lw), self.xs))\
                + "|" + str.center(self.title, rw) + "\n"
        formatted += "+".join(["-"*lw]*len(self.xs)) + "+" + "-"*rw + "\n"
        for i in product(*map(lambda x: range(self.xsize(x)), self.xs)):
            formatted += "|".join(map(lambda i: str.center(str(i), lw), i))
            p = self.lookup(self.xs, i)
            formatted += "|" + str.center(("%." + str(prec) + "f") % p, rw) + "\n"
        return formatted[0:-1]

# 推論結果を持たせる為のクラス
class Report:
    def __init__(self, factor, Q, C, E):
        self.factor = factor
        self.Q = Q
        self.C = C
        self.E = E
        self.prec = 3

class BayesNet:
    def __init__(self):
        self.xs = []
        self.sz = {}
        self.cpts  = []

        # グラフ構造
        self.pa = {}
        self.ch = {}

    # 条件付き確率表 (conditional probability table, CPT) を追加
    # x  : 変数名
    # n  : この変数が取る値の個数
    # pa : 親変数の集合(リスト)
    # tbl: 条件付きテーブル
    def addCPT(self, x, n, pa, tbl):
        self.xs.append(x)
        self.sz[x] = n
        xs = pa + [x]
        sz = map(lambda x: self.sz[x], xs)
        self.cpts.append( Factor(xs, sz, tbl) )
        self.pa[x] = pa
        self.ch[x] = []
        for p in pa:
            self.ch[p].append(x)

    # ファクター集合にエビデンス(x=v,..)を反映して返す.
    @classmethod
    def apply_evidences(cls, factors, evidences):
        for i in range(len(factors)):
            f = factors[i]
            for x in evidences:
                f = f.apply_evidence(x, evidences[x])
            factors[i] = f
        return factors

    # ファクター集合から変数xを消去して, 新しいファクター集合を返す.
    @classmethod
    def eliminate_var(cls, factors, x):
        tobe_removed = [] # 消去するファクター
        not_removed  = [] # 消去されないファクター
        for f in factors:
            if x in f.xs:
                tobe_removed.append(f)
            else:
                not_removed.append(f)

        # 全て掛けてxに関して足し合わせる
        new_factor = reduce(lambda x,y: x*y, tobe_removed).marginalize(x)
        return not_removed + [new_factor]

    # 同時分布 p(Q|E) を計算
    # Q: クエリ集合
    # E: エビデンス集合
    def joint_prob(self, factors, Q, E, removed):
        if Q == []: return 1
        for x in self.xs:
            if x in Q: continue
            if x in removed: continue
            factors = BayesNet.eliminate_var(factors, x)
        factor = reduce(lambda x,y: x*y, factors)
        return factor.normalize()

    # 条件付き確率 p(Q|C,E) を計算
    # Q: クエリ集合
    # C: 条件集合
    # E: エビデンス集合
    def cond_prob(self, factors, Q, C, E, removed):
        # p(Q, C|E) を計算
        pQC = self.joint_prob(factors, Q + C, E, removed)
        if C != []:
            pC  = pQC
            for x in Q:
                pC = pC.marginalize(x)
            pC = pC.normalize()
        else:
            pC = 1
        return pQC/pC

    @classmethod
    def remove_leaf(cls, factors, x):
        return [f for f in factors if not (x in f.xs)]

    @classmethod
    def remove_evidence_node(cls, factors, x, v):
        new_factors = []
        for f in factors:
            f = f.remove_evidence_node(x, v)
            if f.xs == []: continue
            new_factors.append(f)
        return new_factors

    def pruning(self, factors, Q, E):
        xs = set(self.xs) - set(Q) - set(E.keys())
        pa = self.pa.copy()
        ch = self.ch.copy()
        removed = []
        for x in E:
            for y in ch[x]:
                i = self.xs.index(y)
                f = factors[i]
                factors[i] = f.remove_evidence_node(x, E[x])
            if pa[x] == []:
                removed.append(x)
                del factors[self.xs.index(x)]
        while True:
            changed = False
            for x in xs:
                if x in removed: continue
                if ch[x] == []:
                    changed = True
                    removed.append(x)
                    factors = BayesNet.remove_leaf(factors, x)
                    del ch[x]
                    for y in ch:
                        if x in ch[y]: ch[y].remove(x)
            if not changed: break
        return factors, removed

    # 推論を行う
    # Q: クエリ集合
    # C: 条件集合
    # E: エビデンス集合
    def query(self, Q, C=[], E={}):
        factors = copy(self.cpts)

        # エビデンスの反映
        factors = BayesNet.apply_evidences(factors, E)
        # 枝刈り
        factors, removed = self.pruning(factors, Q+C, E)
        # 推論
        f = self.cond_prob(factors, Q, C, E, removed)

        # ファクターの名前を付ける.
        s = "p(" + ",".join(Q)
        if C != []:
            s += "|" + ",".join(C)
        if E != {}:
            if C == []:
                s += "|"
            else:
                s += ","
            s += ",".join(map(lambda x: x + "=" + str(E[x]), E))
        s += ")"
        f.title = s
        return f


sample_net = BayesNet()
sample_net.addCPT('A', 2, [], [0.6, 0.4])
sample_net.addCPT('B', 2, ['A'], [0.2, 0.8, 0.75, 0.25])
sample_net.addCPT('C', 2, ['A'], [0.8, 0.2, 0.1, 0.9])
sample_net.addCPT('D', 2, ['B', 'C'], [0.95, 0.05, 0.9, 0.1, 0.8, 0.2, 0.0, 1.0])
sample_net.addCPT('E', 2, ['C'], [0.7, 0.3, 0.0, 1.0])

print sample_net.query(Q = ['D'], E = {'A':0})
print sample_net.query(Q = ['D'], E = {'E':1})
print sample_net.query(Q = ['B'], E={'D':0})
print sample_net.query(Q = ['B'], E={'A':0, 'D':0})
print sample_net.query(Q = ['B'], E={'A':1, 'D':0})
print sample_net.query(Q = ['B'], C = ['C'], E={'A':0})
print sample_net.query(Q = ['B'], C = ['C'], E={'A':0, 'D':0})
print sample_net.query(Q = ['A', 'B', 'C', 'D', 'E'])
