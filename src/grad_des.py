# Compute the gradient
def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = - tx.T.dot(err) / len(err)

    return grad

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws