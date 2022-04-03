# Planeswalker

For any gradient-based optimization task, a region of configuration space is less likely to contain
a local minimum for two goal functions than it is to contain a local minimum for either goal
function taken separately. In other words: it helps to look at the same problem from many different
but equally valid perspectives.

Therefore, it is better to compose many goal functions together than it is to use only one. The more
"symphonic" your composite goal function is the more likely you are to approach a global minimum so
long as each component of your SGF (symphonic goal function) is harmonious with all other
components. Harmonious meaning: to be different from all other parts in a meaningful way while
sharing one or more global minima with all other parts -- Same global minimum, different local
minima.

Assuming the density of local minima for all of my chosen goal functions are roughly equivalent to,
or cluster around some average value: p, then the density of local minima common to all components
decreases exponentially like p^n, according to the number of harmonious goal functions in the
composition.

Note: multiplexing loss-landscapes is not as simple as summing them together. In fact, any linear
combination of loss-landscapes will only multiply the density of local minima by n, which is
counter-productive. In order to take advantage of the harmony between CGFs (component goal
functions), it is necessary to modulate between them dynamically throughout the optimization
procedure, "paying more attention" to one as the other bottoms out, and to prevent any information
about this modulation action from reaching the optimizer through the optimization variable (the final
number that we are trying to minimize). The best way I've found to accomplish this so far is to
calculate a linear combination of all CGFs weighted by the gradient magnitude of each CGF, and then
subtracting the effect of this modulation (re-weighting) from the optimization variable at each
step. This seems to perform rather well (beats state of the art in some cases) leaving much room for
improvement.

This is something that I have yet to formalize mathematically but I have implimented the theory many
ways, and brought it up to something that, while not nearly perfect, at least makes perfect sense.

The latest test of this idea (and many others frankly, I need to clean this up a lot) is included in
this repo. These tests so far seem extremely promising, as this method of loss-multiplexing succeeds
at certain complex tasks (for example: compressing CIFAR10 by 10x with a VAE, or any kind of images
at all with strong constraints of the batch statistics of the latent space) where vanilla
state-of-the-art methods fail completely.

One good way to demonstrate this theory mathematically would be to take many patches of Perlin noise
(as toy-models of different loss landscapes that would in practice be defined by their own
goal-functions) and align them so that their global minima sit atop each other. How quickly does the
composite get smoother outside the global minima as layers are added? How smooth does the composite
get when the layers are not summed, but instead the layer with the largest gradient at each point is
chosen for the value of that point? Etc.
