'''Testing for the haze model addition.'''
import pdb
from source import hazes

# Constants/inputs
haze_xsecs = "../soot_props/soot.json"
haze_profs = ""

haze = hazes.Haze(haze_profs, haze_xsecs)

pdb.set_trace()
