#!/usr/bin/env python3
import json

with open('game.json', 'r') as f:
    game = json.load(f)

print("First 20 nodes:")
for node in game['nodes'][:20]:
    print(f"Node {node['id']}: {node['kind']}", end="")
    if node['kind'] == 'decision':
        print(f" player={node['player']} infoset={node['infoset']} actions={[a['label'] for a in node['actions']]}")
    elif node['kind'] == 'terminal':
        print(f" payoff_p0={node['payoff_p0']}")
    elif node['kind'] == 'chance':
        print(f" outcomes={len(node['outcomes'])}")

print("\nAll information sets:")
infosets = {}
for node in game['nodes']:
    if node['kind'] == 'decision':
        infoset = node['infoset']
        if infoset not in infosets:
            infosets[infoset] = (node['player'], [a['label'] for a in node['actions']])

for infoset, (player, actions) in sorted(infosets.items()):
    print(f"  {infoset}: P{player}, actions={actions}")
