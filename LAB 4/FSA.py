from graphviz import Digraph

FSA = Digraph(name='FSA')
FSA.attr(rankdir='LR')
FSA.node('A', 'q0', shape='circle')
FSA.node('B', 'q1', shape='circle')
FSA.node('L', 'q2', shape='doublecircle')

FSA.edge('A', 'A', label='b')
FSA.edge('A', 'B', label='a')
FSA.edge('B', 'A', label='b')
FSA.edge('B', 'L', label='a')
FSA.edge('L', 'A', label='b')
FSA.edge('L', 'L', label='a')

print(FSA.source)
FSA.view() # FSA.render('test-output/round-table.gv', view=True)