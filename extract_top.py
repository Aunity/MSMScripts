import os
import sys
import mdtraj as md
def main():
    if len(sys.argv[1:])!=3:
        print("Usage: python %s <pdbfile> <n> <name>"%sys.argv[0])
        exit(0)
    pdbfile, n, name = sys.argv[1:]
    pdb = md.load_pdb(pdbfile)
    for i in range(int(n)):
        pdb[i].save_pdb(name+'-%d.pdb'%i)
        print(i)
if __name__ == "__main__":
    main()
