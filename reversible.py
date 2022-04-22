from functools import partial
import jax
import jax.numpy as np

# https://jax.readthedocs.io/en/latest/custom_vjp_update.html
# https://github.com/google/jax/issues/2912

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def my_fori_loop(n, f, f_inv, x):
    return jax.lax.fori_loop(0, n, (lambda _, x: f(x)), x)

def my_fori_loop_fwd(n, f, f_inv, x):
    y = my_fori_loop(n, f, f_inv, x)
    return y, y

def my_fori_loop_rev(n, f, f_inv, y, o):
    def step(yo):
        y, o = yo
        x = f_inv(y)
        _, f_vjp = jax.vjp(f, x)
        return x, f_vjp(o)[0]

    y_, o_ = jax.lax.fori_loop(
        0, n,
        (lambda _, yo: step(yo)),
        (y, o)
    )
    return (o_,)

my_fori_loop.defvjp(my_fori_loop_fwd, my_fori_loop_rev)