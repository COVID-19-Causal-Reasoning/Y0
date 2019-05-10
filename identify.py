
import networkx as nx
from matplotlib import pylab as plt
import seaborn as sns
import inspect
from collections import OrderedDict
import pgmpy as pgm
from pgmpy.models import BayesianModel
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.factors.discrete import TabularCPD, JointProbabilityDistribution
from pgmpy.base import DAG
from pgmpy.extern.six.moves import reduce
import numpy as np
import pandas as pd
from numpy.random import randn
from itertools import product
from pgmpy.factors.base import factor_product
from IPython.display import Latex
import sys
from matplotlib.patches import FancyArrowPatch, Circle 
from sympy import Function, Symbol, Eq, simplify, init_printing, latex
from sympy.concrete.summations import Sum
from sympy.core.core import all_classes as sympy_classes
from sympy.concrete.products import Product
from sympy.functions import Abs
from IPython.display import display, Latex
class Fail(Exception):
    pass



def draw_graph( G, U ):
    pos = nx.spring_layout(G)
    V = G.nodes
    #nx.draw(G, pos)
    
    nx.draw_networkx_nodes(G, pos,
                       nodelist=V,
                       node_color='r',
                       node_size=500,
                       alpha=0.5)
    nx.draw_networkx_edges(U, pos,
                       edgelist=list(U.edges),
                       linestyle='dashed',
                       edge_color='green',
                       arrows=True,
                       arrowstyle='<|-|>',
                       connectionstyle="arc3,rad=0.3",
                       width=3, alpha=0.5)
    nx.draw_networkx_edges(G, pos,
                          edgelist=list(G.edges),
                          edge_color='black',
                          style='solid',
                          width=2 )
    nx.draw_networkx_labels(G, pos)
    plt.show()

def to_frame(self, tablefmt="fancy_grid", print_state_names=True):
        headers_list = []
        # build column headers

        evidence = self.variables[1:]
        evidence_card = self.cardinality[1:]
        if evidence:
            col_indexes = np.array(list(product(*[range(i) for i in evidence_card])))
            if self.state_names and print_state_names:
                for i in range(len(evidence_card)):
                    column_header = [str(evidence[i])] + ['{var}({state})'.format
                                                     (var=evidence[i],
                                                      state=self.state_names[evidence[i]][d])
                                                     for d in col_indexes.T[i]]
                    headers_list.append(column_header)
            else:
                for i in range(len(evidence_card)):
                    column_header = [str(evidence[i])] + ['{s}_{d}'.format(s=evidence[i], d=d) 
                                                          for d in col_indexes.T[i]]
                    headers_list.append(column_header)

        # Build row headers
        if self.state_names and print_state_names:
            variable_array = [['{var}({state})'.format
                               (var=self.variable, state=self.state_names[self.variable][i])
                               for i in range(self.variable_card)]]
        else:
            variable_array = [['{s}_{d}'.format(s=self.variable, d=i) for i in range(self.variable_card)]]
        # Stack with data
        labeled_rows = np.hstack((np.array(variable_array).T, self.get_values())).tolist()
        # No support for multi-headers in tabulate
        if len(headers_list) > 0:
            column_idx = pd.MultiIndex.from_arrays([header[1:] for header in headers_list],
                                              names= [header[0] for header in headers_list])
            row_idx_name = 'P({}|{})'.format(self.variable, ','.join(column_idx.names))
        else:
            column_idx = [self.variable]
            row_idx_name = 'P({})'.format(self.variable)
        df = pd.DataFrame(self.get_values(), columns=column_idx, index=variable_array)
        df.index.name = row_idx_name
        return df

def adjacent_nodes( pairs, node ):
    "Returns the nodes adjacent to node (via the bidirected edges in pairs)."
    adjacent_nodes = set()
    for pair in pairs:
        if node in pair:
            adjacent_nodes |= set(pair)
    return list(adjacent_nodes - set([node]))
   
def factorize_c_components( U ):
    """Returns the confounded components of G as a list of sets of vertices"""
    def recur(nodes = list(U.nodes), components = list()):
        if len(nodes) == 0:
            return components
        else:
            current_node = nodes[0]
            current_component = connected_component( U.edges ,
                                                     current_node )
            print('found connected component {} of node {}'.format(current_component, current_node))
            
            return recur( list(set(nodes[1:]) - current_component ),
                          components + [current_component] ) 
    return recur()
def connected_component( pairs, node ):
    def recur( frontier=[node], visited=set() ):
        if len(frontier) == 0:
            return visited
        else:
            current = frontier[0]
            if current in visited:
                return recur( frontier[1:], visited )
            else:
                visited.add( current )
                return recur( frontier[1:] + adjacent_nodes( pairs, current),
                              visited )
    return recur( )


def remove_nodes_from( G , x ):
    G_x = G.copy()
    G_x.remove_nodes_from( x )
    return G_x

def predecessors( pi, v ):
    return list(pi)[:pi.index(v)]

def marginalize( marginals, P ):
    if type(P) is list:
        return pgmpy_marginalize( marginals, P )
    if type(P) in sympy_classes:
        return sympy_marginalize( marginals, P )
    
def sympy_marginalize( marginals, P ):
    for marginal in marginals:
        V = Symbol(marginal.upper())
        if V in P.free_symbols:
            v = Symbol(str(V).lower())
            P = Sum(P, (V, v, Abs(V)) )
        else:
            print('Marginal {} is not a free variable in P: {}'.format(V, P.free_symbols))
    return P
def pgmpy_marginalize( marginals, P ):
    return [Pv.marginalize( (set(Pv.scope()) & marginals) - set([Pv.variable]),
                          inplace=False)
           for Pv in P]

def predecessors( pi, vi ):
    return pi[:pi.index(vi)]
def sympy_given( P, vi, pi ):
    pred    = predecessors( pi, vi )
    unbound = set([str(v) for v in P.free_symbols]) - (set([vi]) | set(pred) )
    numer   = marginalize( unbound, P )
    denom   = marginalize( set([vi]), numer )
    return numer/denom

def given( P, vi, pi ):
    if type( P ) in sympy_classes:
        return sympy_given( P, vi, pi )
    elif type( P ) is str:
        return pgmpy_given( P, vi, pi )
    
        
def sum_product( marginals, P ):
    if type(P) is list:
        return pgmpy_sum_product( marginals, P )
    elif type(P) in sympy_classes:
        return sympy_sum_product( marginals, P )

def sympy_product( P_list ):
    P_product = P_list[0]
    if len(P_list) > 1:
        for P in P_list[1:]:
            P_product *= P
    return P_product

def product( P_list ):
    if all([type(P) is list for P in P_list]):
        return pgmpy_product( P_list )
    elif all([type( P ) in sympy_classes for P in P_list]):
        return sympy_product( P_list )

def pgmpy_product( P_list ):
    P_product = P_list[0]
    if len(P_list) > 1:
        return factor_product( *P )
    elif len(P_list) == 1:
        return P_product
    else:
        raise NameError("P is empty!")
def sympy_sum_product( marginals, P_list ):
    return marginalize( marginals, product( P_list ) )
    

def pgmpy_sum_product(  marginals, P ):
    if len(P) > 1:
        return factor_product( *P ).\
               marginalize( marginals, inplace=False )
    elif len(P) == 1:
        return P[0].marginalize( marginals, inplace=False )
    else:
        raise NameError("P is empty!")
        
def joint_probability_distribution( vertices ):
    P = Function('P')
    return P(*[Symbol(V.upper()) for V in vertices])

def cut_incoming( G, x ):
    G_x = G.copy()
    for edge in G.edges:
        if edge[1] in x:
            G_x.remove_edge( *edge )
    return G_x
def draw_bidirected_graph( bigraph ):
    pos = nx.spring_layout(bigraph)
    nx.draw(bigraph, pos)
    nx.draw_networkx_labels(bigraph, pos)
    plt.show()

def find_superset( C_component, s ):
    for s_prime in C_component:
        if len(s - s_prime) == 0:
            return s_prime
    return []

def display_P( P ):
    if type(P) is list:
        return [display(to_frame(cpd)) for cpd in P]
    elif type(P) in sympy_classes:
        return display(P)
def ID( y, x, P, G, U ):
    display(Latex('Identify $P({} | {})$'.format(','.join([latex(Symbol(yi)) for yi in sorted(y)]), 
                                        ','.join(['do({})'.format(latex(Symbol(xi)))
                                                  for xi in sorted(x)]))))
    display_P( P) 
    draw_graph( G, U)
    v = set(G.nodes)
    # line 1
    if len(x) == 0:
        print('Line 1')
        return marginalize( v - y, P )
    # line 2
    ancestors_y = G._get_ancestors_of( list(y) )
    if len(v - ancestors_y) > 0:
        print('Line 2')
        ID( y, 
            x & ancestors_y,
           marginalize( v - ancestors_y, P ),
           G.subgraph( ancestors_y ),
           U.subgraph( ancestors_y )
          )
    
    # line 3
    G_bar_x = cut_incoming( G, x )
    w = (v - x) - G_bar_x._get_ancestors_of( list(y) )
    if len( w ) > 0:
        print('Line 3')
        ID( y, x | w, P, G, U )
        
    # line 4
    U_x = remove_nodes_from( U, x )
    C_components_of_U_x = factorize_c_components( U_x )
    print('C_x: {}'.format(C_components_of_U_x))
    if len(C_components_of_U_x) > 1:
        print('Line 4')
        return marginalize( v - (x|y), 
                            product([ID(  C_component, 
                                  v - C_component,
                                  P,
                                  G,
                                  U )
                            for C_component in C_components_of_U_x] ))

    
    else:
        # line 5
        print('C-component of U_x: {}'.format(C_components_of_U_x))
        if len(C_components_of_U_x) == 1:
            S_x = C_components_of_U_x[0]
        else:
            S_x = set()
        C_components_of_U = factorize_c_components( U )
        print('C-component of U: {}'.format(C_components_of_U))
        print('Is C(U)={} == U={}: {}'.format( C_components_of_U[0], set(U.nodes), C_components_of_U[0] == set(U.nodes)))
        if len(C_components_of_U) == 1 and C_components_of_U[0] == set(U.nodes):
            print('Line 5')
            raise Fail( 
                "C_components_of_U {} and C_components_of_U_x {} form a hedge".format(
                    C_components_of_U, C_components_of_U_x ))

        # line 6
        pi = list(nx.topological_sort( G ))
        if S_x  in C_components_of_U:
            print('Line 6')
            return marginalize(S_x - y, 
                               product([given( P, vi, pi ) 
                                         for vi in S_x] ))
        # line 7
        S_prime = find_superset( C_components_of_U, S_x )  
        if len(S_prime) > 0:
            print('Line 7')
            print('Found superset: {}'.format( S_prime ))
            P_prime = product([given( P, vi, pi )
                               for vi in S_prime])
            return ID( y,
                       x & S_prime,
                       P_prime,
                       G.subgraph( S_prime ), 
                       U.subgraph( S_prime )
                     )
        else:
            raise Error("S' is empty")