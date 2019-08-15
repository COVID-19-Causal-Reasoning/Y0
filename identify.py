import networkx as nx
from matplotlib import pylab as plt
import seaborn as sns
import inspect
from collections import OrderedDict
from pgmpy.base import DAG
from pgmpy.extern.six.moves import reduce
import numpy as np
import pandas as pd
from numpy.random import randn
from itertools import product
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
            #print('found connected component {} of node {}'.format(current_component, current_node))
            
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
    V = P.free_symbols - set([Symbol(m.upper()) for m in marginals])
    return joint_probability_distribution( V )

def old_sympy_marginalize( marginals, P ):
    for marginal in marginals:
        V = Symbol(marginal.upper())
        if V in P.free_symbols:
            v = Symbol(str(V).lower())
            P = Sum(P, (v, 1, Abs(V)) )
        else:
            print('Marginal {} is not a free variable in P: {}'.format(V, P.free_symbols))
    return P

def predecessors( pi, vi ):
    return pi[:pi.index(vi)]
def given( P, vi, pi ):
    pred    = predecessors( pi, vi )
    unbound = set([str(v) for v in P.free_symbols]) - (set([vi]) | set(pred) )
    numer   = marginalize( unbound, P )
    denom   = marginalize( set([vi]), numer )
    return numer/denom

        

def product( P_list ):
    P_product = P_list[0]
    if len(P_list) > 1:
        for P in P_list[1:]:
            P_product *= P
    return P_product


def sum_product( marginals, P_list ):
    return old_sympy_marginalize( marginals, product( P_list ) )
    

def joint_probability_distribution( vertices ):
    P = Function('P')
    return P(*[Symbol(str(V).upper()) for V in vertices])

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
def ID( y, x, P, G, U, debug=False, recurse_level = 0 ):
    if debug:
        display(Latex('{}Identify $P({} | {})$'.format('-'*recurse_level,','.join([latex(Symbol(yi)) for yi in sorted(y)]), 
                                        ','.join(['do({})'.format(latex(Symbol(xi)))
                                                  for xi in sorted(x)]))))
        display_P( P) 
        draw_graph( G, U)
    v = set(G.nodes)
    # line 1
    if len(x) == 0:
        if debug:
            print('{}Line 1'.format('-'*recurse_level))
            display(Latex('{}No $do(X)$. Return ${}$'.format('-'*recurse_level,  latex(marginalize( v - y, P)))))
        return marginalize( v - y, P )
    # line 2
    ancestors_y = set(G._get_ancestors_of( list(y) ))
    
    if len(v - ancestors_y) > 0:
        if debug:
            print('{}Line 2'.format('-'*recurse_level))
            print('{}Removing non-ancestors of Y={}:  {}'.format('-'*recurse_level, y,  v - ancestors_y ))
        return ID( y, 
                   x & ancestors_y,
                   marginalize( v - ancestors_y, P ),
                   G.subgraph( ancestors_y ),
                   U.subgraph( ancestors_y ),
                   debug = debug,
                   recurse_level = recurse_level + 1
          )
    
    # line 3
    G_bar_x = cut_incoming( G, x )
    w = (v - x) - G_bar_x._get_ancestors_of( list(y) )
    if len( w ) > 0:
        if debug:
            print('{}Line 3'.format('-'*recurse_level))
        return ID( y, x | w, P, G, U, debug=debug, recurse_level = recurse_level + 1 )
        
    # line 4
    U_x = remove_nodes_from( U, x )
    G_x = remove_nodes_from( G, x )
    
    C_components_of_U_x = factorize_c_components( U_x )
    if debug:
        print('{}C_x: {}'.format('-'*recurse_level,C_components_of_U_x))
    if len(C_components_of_U_x) > 1:
        if debug:
            print('{}Line 4'.format('-'*recurse_level))
            display('{}G - X={}:'.format('-'*recurse_level,x))
        draw_graph( G_x, U_x)
        P_list = []
        for C_component in C_components_of_U_x:
            Ps = ID( C_component, v - C_component, P, G, U, debug=debug, recurse_level=recurse_level + 1 )
            if debug:
                display(Latex('{}\tC-component Identify $P({} | {}) = {}$'.format(
                              '-'*recurse_level,
                              ','.join([latex(Symbol(yi)) 
                                  for yi in sorted(C_component)]), 
                              ','.join(['do({})'.format(latex(Symbol(xi)))
                                     for xi in sorted(v - C_component)]),
                               latex(Ps))))
            
            P_list.append( Ps )
        if debug:
            display('{}Returning back to Line 4'.format('-'*recurse_level))
        return marginalize( v - (x|y), 
                            product(P_list ))

    
    else:
        # line 5
        if debug:
            print('{}C-component of U_x: {}'.format('-'*recurse_level, C_components_of_U_x))
        if len(C_components_of_U_x) == 1:
            S_x = C_components_of_U_x[0]
        else:
            S_x = set()
        C_components_of_U = factorize_c_components( U )
        if debug:
            print('{}C-component of U: {}'.format('-'*recurse_level, C_components_of_U))
            print('{}Is C(U)={} equal to U={}: {}'.format('-'*recurse_level,  C_components_of_U[0], set(U.nodes), C_components_of_U[0] == set(U.nodes)))
        if len(C_components_of_U) == 1 and C_components_of_U[0] == set(U.nodes):
            if debug:
                print('{}Line 5'.format('-'*recurse_level))
            raise Fail( 
                "{}Identification Failure: C-components of U {} and C-components of (U-x) {} form a hedge".format(
                    '-'*recurse_level, C_components_of_U, C_components_of_U & S_x ))

        # line 6
        pi = list(nx.topological_sort( G ))
        if S_x  in C_components_of_U:
            if debug:
                print('{}Line 6'.format('-'*recurse_level))
            return marginalize(S_x - y, 
                               product([given( P, vi, pi )  # P( vi | pi )
                                         for vi in S_x] ))
        # line 7
        S_prime = find_superset( C_components_of_U, S_x )  
        if len(S_prime) > 0:
            if debug:
                print('{}Line 7'.format('-'*recurse_level))
                print('{}Found superset: {}'.format( '-'*recurse_level, S_prime ))
            P_prime = product([given( P, vi, pi )
                               for vi in S_prime])
            return ID( y,
                       x & S_prime,
                       P_prime,
                       G.subgraph( S_prime ), 
                       U.subgraph( S_prime ), 
                       debug = debug,
                       recurse_level = recurse_level + 1
                     )
        else:
            raise Error("{}S' is empty".format('-'*recurse_level))
