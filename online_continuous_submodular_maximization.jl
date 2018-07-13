using MathProgBase, Ipopt, StatsBase, PyPlot, LaTeXStrings, Base.Iterators.cycle, Clp
PyPlot.matplotlib[:rcParams]["figure.autolayout"] = "True"
srand(0)
PyPlot.matplotlib[:rc]("font", family="serif", serif="Times", size=16)
global_width = 3
type RFTL
    x
    eta
    projection
end
function update(rftl, gradient)
    rftl.x = rftl.projection(rftl.x + rftl.eta * gradient)
end
function get_vector(rftl)
    return rftl.x
end
function play_sound()
    if is_apple()
        run(`say "your program has finished"`)
    end
end
function my_show()
    if is_apple()
        show()
    end
end
function gibbs_sampler(A, b, x)
    n = size(A)[2]
    for iter in 1:n * 3
        j = rand(1:n)
        x_copy = x
        x_copy[j] = 0
        max_coord = minimum((b - A * x_copy) ./ A[:, j])
        max_coord = min(max_coord, 1)
        x[j] = rand() * max_coord
    end
    return x
end
function meta_frank_wolfe(dim_lam, X_list, projection, f, gradient, eta, T, K)
    x0 = zeros(dim_lam)
    rftl = [RFTL(x0, eta, projection) for k in 1:K]
    reward = []
    for t in 1:T
        X = X_list[t]
        v = [get_vector(rftl[k]) for k in 1:K]
        x = zeros(K + 1, dim_lam)
        for k in 1:K
            update(rftl[k], gradient(x[k, :], X))
            x[k + 1, :] = x[k, :] + v[k] / K
        end
        push!(reward, f(x[K + 1, :], X))
    end
    reward = cumsum(reward)
    return reward
end
function frank_wolfe(dim_lam, X, A, b, f, gradient, K)
    x = zeros(dim_lam)
    for k in 1:K
        grad_x = gradient(x, X)
        sol = linprog(-grad_x, A, '<', b, 0.0, 1.0, ClpSolver())
        if sol.status == :Optimal
            v = sol.sol
        else
            error("No solution was found.")
        end
        x += v / K
    end
    return x
end
function frank_wolfe2(dim_lam, A, b, gradient, K)
    x = zeros(dim_lam)
    for k in 1:K
        grad_x = gradient(x)
        sol = linprog(-grad_x, A, '<', b, 0.0, 1.0, ClpSolver())
        if sol.status == :Optimal
            v = sol.sol
        else
            error("No solution was found.")
        end
        x += v / K
    end
    return x
end
function online_gradient_ascent(dim_lam, X_list, projection, f, gradient, eta, T)
    x0 = zeros(dim_lam)
    rftl = RFTL(x0, eta, projection)
    reward = []
    for t in 1:T
        X = X_list[t]
        v = get_vector(rftl)
        push!(reward, f(v, X))
        update(rftl, gradient(v, X))
    end
    reward = cumsum(reward)
    return reward
end
function online_random(dim_lam, X_list, f, linear_projection, T)
    reward = []
    for t in 1:T
        X = X_list[t]
        v = rand(dim_lam)
        v = linear_projection(v)
        push!(reward, f(v, X))
    end
    reward = cumsum(reward)
    return reward
end
function online_random_gibbs(dim_lam, X_list, f, A, b, T)
    reward = zeros(T)
    gibbs_v = zeros(size(A)[2])
    for t in 1:T
        X = X_list[t]
        for j in 1:100
            gibbs_v = gibbs_sampler(A, b, gibbs_v)
            reward[t] = max(reward[t], f(gibbs_v, X))
        end
    end
    reward = cumsum(reward)
    return reward
end
function generate_projection_function(A, b)
    n = size(A)[2]
    function projection(x0)
        sol = quadprog(-x0, eye(n), A, '<', b, 0.0, 1.0, IpoptSolver(print_level=0))
        if sol.status == :Optimal
            return sol.sol
        end
        error("No solution was found.")
    end
    return projection
end


function offline(X_list, f, gradient, A, b, K)
    T = length(X_list)
    reward = zeros(T)
    gradient_cumul = x->sum([gradient(x, params) for params in X_list])
    v = frank_wolfe2(size(A)[2], A, b, gradient_cumul, K)
    for i in 1:T
        reward[i] = f(v, X_list[i])
    end
    return cumsum(reward)
end


function experiment_coverage(size_U, n_sets, T, K)
    function generate_a_param()
        B = []
        for i in 1:n_sets
            lo = round(Int, size_U * 0.2)
            hi = round(Int, size_U * 0.8)
            set_size = rand(lo:hi)
            push!(B, IntSet(sample(1:size_U, set_size, replace=false)))
        end
        w = rand(size_U) * 100
        return (B, w)
    end
    X_list = [generate_a_param() for i in 1:T]
    function f_cover(x, params)
        B = params[1]
        w = params[2]
        res = 0
        for u in 1:size_U
            prod = 1
            for i in 1:n_sets
                if u in B[i]
                    prod *= (1 - x[i])
                end
            end
            res += w[u] * (1 - prod)
        end
        return res
    end
    function concave_f(x)
        res = 0
        for u in 1:size_U
            res += w[u] * min(1, sum([x[i] for i in xrange(n_sets) if u in B[i]]))
        end
        return res
    end
    function partial(x, i, B, w)
        res = 0
        for u in 1:size_U
            prod = 1
            for j in 1:n_sets
                if j != i && u in B[j]
                    prod *= (1 - x[j])
                end
            end
            res += w[u] * prod
        end
        return res
    end
    function gradient_cover(x, params)
        B = params[1]
        w = params[2]
        res = zeros(n_sets)
        for i in 1:n_sets
            res[i] = partial(x, i, B, w)
        end
        return res
    end
    function stochastic_gradient_cover(x, params)
        B = params[1]
        w = params[2]
        function func(S)
            if isempty(S)
                return 0
            end
            all_sets = [B[i] for i in S]
            union_set = reduce(union, all_sets)
            return sum([w[i] for i in union_set])
        end
        function gradient_i(i)
            R = [j for j in 1:n_sets if j != i && rand() < x[j]]
            R_i = vcat(R, [i])
            return func(R_i) - func(R)
        end
        grad = zeros(n_sets)
        for i in 1:n_sets
            grad[i] = gradient_i(i)
        end
        return grad
    end
    function gradient_concave(x, params)
        B = params[1]
        w = params[2]
        function partial_concave(i)
            res = 0
            for u in 1:size_U
                if !(u in B[i])
                    continue
                end
                if sum([x[j] for j in 1:n_sets if u in B[j]]) <= 1
                    res += w[u]
                end
            end
            return res
        end
        grad = zeros(n_sets)
        for i in 1:n_sets
            grad[i] = partial_concave(i)
        end
        return grad
    end
    n_constraints = 2
    A = rand(n_constraints, n_sets)
    b = 1.0
    linear_projection = generate_projection_function(A, b)
    reward_offline = offline(X_list, f_cover, gradient_cover, A, b, K)
    println(reward_offline)
    function generate_figure_regret()
        reward_list = []
        push!(reward_list, reward_offline - meta_frank_wolfe(n_sets, X_list,
                                                    linear_projection, f_cover, gradient_cover, 0.1, T, K))
        push!(reward_list, reward_offline - meta_frank_wolfe(n_sets, X_list,
                                                    linear_projection, f_cover, gradient_cover, 0.01, T, K))
        push!(reward_list, reward_offline - online_gradient_ascent(n_sets, X_list,
                                                linear_projection, f_cover, gradient_cover, 0.1, T))
        push!(reward_list, reward_offline - online_gradient_ascent(n_sets, X_list,
                                                linear_projection, f_cover, gradient_cover, 0.01, T))
        push!(reward_list, reward_offline - online_gradient_ascent(n_sets, X_list,
                                                linear_projection, f_cover, gradient_concave, 0.1, T))
        push!(reward_list, reward_offline - online_gradient_ascent(n_sets, X_list,
                                                linear_projection, f_cover, gradient_concave, 0.01, T))
        push!(reward_list, reward_offline - online_random_gibbs(n_sets, X_list, f_cover, A, b, T))
        labels = [L"FW ($\eta=0.1$)", L"FW ($\eta=0.01$)", L"GA ($\eta=0.1$)", L"GA ($\eta=0.01$)", L"SurrGA ($\eta=0.1$)", L"SurrGA ($\eta=0.01$)", "OfflApprMax"]        
        marker = cycle((".", ",", "+", "_", "o", "x", "*"))
        linecycler = cycle(("-", "--", "-.", ":"))
        println(reward_list)
        for zipped in zip(reward_list, labels, marker, linecycler)
            reward, label, marker_iter, line_iter = zipped
            plot(1:T, reward, label=label, linestyle=line_iter, linewidth=global_width)
        end
        ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        legend(loc="best")
        xlabel("Iteration index")
        ylabel(L"$(1-1/e)$-Regret")
        grid()       
        # savefig("coverage_regret_T2.pdf")
        my_show()
    end
    function generate_figure_regret_stochastic()
        reward_list = []
        push!(reward_list, reward_offline - meta_frank_wolfe(n_sets, X_list,
                                                    linear_projection, f_cover, stochastic_gradient_cover, 0.1, T, K))
        push!(reward_list, reward_offline - meta_frank_wolfe(n_sets, X_list,
                                                    linear_projection, f_cover, stochastic_gradient_cover, 0.01, T, K))
        push!(reward_list, reward_offline - online_gradient_ascent(n_sets, X_list,
                                                linear_projection, f_cover, stochastic_gradient_cover, 0.1, T))
        push!(reward_list, reward_offline - online_gradient_ascent(n_sets, X_list,
                                                linear_projection, f_cover, stochastic_gradient_cover, 0.01, T))
        push!(reward_list, reward_offline - online_gradient_ascent(n_sets, X_list,
                                                linear_projection, f_cover, gradient_concave, 0.1, T))
        push!(reward_list, reward_offline - online_gradient_ascent(n_sets, X_list,
                                                linear_projection, f_cover, gradient_concave, 0.01, T))
        push!(reward_list, reward_offline - online_random_gibbs(n_sets, X_list, f_cover, A, b, T))
        labels = [L"FW ($\eta=0.1$)", L"FW ($\eta=0.01$)", L"GA ($\eta=0.1$)", L"GA ($\eta=0.01$)", L"SurrGA ($\eta=0.1$)", L"SurrGA ($\eta=0.01$)", "OfflApprMax"]        
        marker = cycle((".", ",", "+", "_", "o", "x", "*"))
        linecycler = cycle(("-", "--", "-.", ":"))
        println(reward_list)
        for zipped in zip(reward_list, labels, marker, linecycler)
            reward, label, marker_iter, line_iter = zipped
            plot(1:T, reward, label=label, linestyle=line_iter, linewidth=global_width)
        end
        ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        legend(loc="best")
        xlabel("Iteration index")
        ylabel(L"$(1-1/e)$-Regret")
        grid()       
        # savefig("stoch_coverage_regret_T2.pdf")
        my_show()
    end
    # generate_figure_regret()
    generate_figure_regret_stochastic()
end
function experiment_nqp(dim_x, T, K)
    function generate_a_param(dim_x, alpha=100)
        return rand(dim_x, dim_x) * (-alpha)
    end
    n_constraints = 2
    A = rand(n_constraints, dim_x)
    b = 1.0
    linear_projection = generate_projection_function(A, b)
    function f_nqp(x, H)
        return ((x / 2 - 1)' * H * x)
    end
    function gradient_nqp(x, H)
        return (H + H') / 2 * x - H' * ones(dim_x)
    end
    function generate_figure_regret()
        all_reward_lists = []
        for experiment_iter in 1:1
            X_list = [generate_a_param(dim_x) for i in 1:T]
            reward_offline = offline(X_list, f_nqp, gradient_nqp, A, b, K)
            reward_list = []
            push!(reward_list, reward_offline - meta_frank_wolfe(dim_x, X_list,
                                                        linear_projection, f_nqp, gradient_nqp, 0.001, T, K))
            push!(reward_list, reward_offline - meta_frank_wolfe(dim_x, X_list,
                                                        linear_projection, f_nqp, gradient_nqp, 0.0008, T, K))
            push!(reward_list, reward_offline - online_gradient_ascent(dim_x, X_list,
                                                    linear_projection, f_nqp, gradient_nqp, 0.001, T))
            push!(reward_list, reward_offline - online_gradient_ascent(dim_x, X_list,
                                                    linear_projection, f_nqp, gradient_nqp, 0.0008, T))
            push!(all_reward_lists, reward_list)
        end
        all_reward_lists = mean(all_reward_lists)
        labels = [L"FW ($\eta=0.1$)", L"FW ($\eta=0.01$)", L"GA ($\eta=0.1$)", L"GA ($\eta=0.01$)"]
        marker = cycle((".", ",", "+", "_", "o", "x", "*"))
        linecycler = cycle(("-", "--", "-.", ":"))
        println(all_reward_lists)
        for zipped in zip(all_reward_lists, labels, marker, linecycler)
            reward, label, marker_iter, line_iter = zipped
            plot(1:T, reward, label=label,  linestyle=line_iter, linewidth=global_width) 
        end
        ticklabel_format(style="sci", axis="y", scilimits=(0, 0))        
        legend(loc="best")
        xlabel("Iteration index")
        ylabel(L"$(1-1/e)$-Regret")
        grid()
        play_sound()
        my_show()
    end
    function generate_figure_regret_dim(alpha)
        dim_x_list = 5:5:50
        all_reward_lists = []
        reward_list = zeros(4, size(dim_x_list)[1])
        for experiment_iter in 1:5
            for (idx, dim_x) in enumerate(dim_x_list)
                tic()
                println("parameter\t", idx, ", ", dim_x)
                X_list = [generate_a_param(dim_x, alpha) for i in 1:T]
                A = rand(n_constraints, dim_x)
                b = 1.0
                linear_projection = generate_projection_function(A, b) 
                X_list_sum = sum(X_list)
                v = frank_wolfe(dim_x, X_list_sum, A, b, f_nqp, gradient_nqp, K)
                reward_offline = f_nqp(v, X_list_sum)
                reward_list[:, idx] =  reward_offline - [meta_frank_wolfe(dim_x, X_list, linear_projection, f_nqp, gradient_nqp, 0.001, T, K)[end],
                    meta_frank_wolfe(dim_x, X_list, linear_projection, f_nqp, gradient_nqp, 0.0008, T, K)[end],
                    online_gradient_ascent(dim_x, X_list, linear_projection, f_nqp, gradient_nqp, 0.001, T)[end],
                    online_gradient_ascent(dim_x, X_list, linear_projection, f_nqp, gradient_nqp, 0.0008, T)[end]]
                toc()
            end
            push!(all_reward_lists, reward_list)
        end
        reward_list = mean(all_reward_lists)
        reward_list = [reward_list[i, :] for i in 1:size(reward_list)[1]]
        labels = [L"FW ($\eta=0.1$)", L"FW ($\eta=0.01$)", L"GA ($\eta=0.1$)", L"GA ($\eta=0.01$)"]
        marker = cycle((".", ",", "+", "_", "o", "x", "*"))
        linecycler = cycle(("-", "--", "-.", ":"))
        print(reward_list)
        for zipped in zip(reward_list, labels, marker, linecycler)
            reward, label, marker_iter, line_iter = zipped
            plot(dim_x_list, reward, label=label, linestyle=line_iter, linewidth=global_width)
        end
        legend(loc="best")
        ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        xlabel(L"Dimension of $\mathbf{x}$")
        ylabel(L"$(1-1/e)$-Regret")
        grid()  
        play_sound()
        my_show()
    end
    generate_figure_regret()
    # generate_figure_regret_dim(1)
end
function experiment_design(dim_lam, dim_x_vectors, T, K)
    function generate_a_param(dim_lam, dim_x_vectors)
        mat = randn(dim_x_vectors, dim_lam) 
        return mat
    end
    X_list = [generate_a_param(dim_lam, dim_x_vectors) for i in 1:T]
    n_constraints = 2
    A = rand(n_constraints, dim_lam)
    b = 1.0
    linear_projection = generate_projection_function(A, b)
    offset_constant = 1
    multp = 1
    function f(lam, X)
        lam += offset_constant
        lam *= multp
        M = 0
        for i in 1:dim_lam
            M += lam[i] * X[:, i] * X[:, i]'
        end
        return log(det(M))
    end
    function gradient(lam, X)
        lam += offset_constant
        lam *= multp
        M = 0
        for i in 1:dim_lam
            M += lam[i] * X[:, i] * X[:, i]'
        end
        grad = zeros(lam)
        for i in 1:size(lam)[1]
            grad[i] = X[:, i]' * inv(M) * X[:, i]
        end
        return grad
    end
    function gradient_orth(lam, X)
        return 1 ./ (lam + offset_constant)
    end
    function f_a(lam, X)
        lam += offset_constant
        M = 0
        for i in 1:dim_lam
            M += lam[i] * X[:, i] * X[:, i]'
        end
        return -trace(inv(M))
    end
    function gradient_a(lam, X)
        lam += offset_constant
        M = 0
        for i in 1:dim_lam
            M += lam[i] * X[:, i] * X[:, i]'
        end
        inv_M = inv(M)
        grad = zeros(lam)
        for i in 1:size(lam)[1]
            grad[i] = norm(inv_M * X[:, i])^2
        end
        return grad
    end
    function generate_figure_regret(f, gradient)
        reward_list = []
        reward_offline = offline(X_list, f, gradient, A, b, K)
        println(reward_offline)
        push!(reward_list, reward_offline - meta_frank_wolfe(dim_lam, X_list,
                                                     linear_projection, f, gradient, 0.8, T, K))
        push!(reward_list, reward_offline - meta_frank_wolfe(dim_lam, X_list,
                                                     linear_projection, f, gradient, 1, T, K))
        push!(reward_list, reward_offline - online_gradient_ascent(dim_lam, X_list,
                                                  linear_projection, f, gradient, 0.8, T))
        push!(reward_list, reward_offline - online_gradient_ascent(dim_lam, X_list,
                                                  linear_projection, f, gradient, 1, T))
        labels = [L"FW ($\eta=200$)", L"FW ($\eta=100$)", L"GA ($\eta=200$)", L"GA ($\eta=100$)", "OfflApprMax"]        
        marker = cycle((".", ",", "+", "_", "o", "x", "*"))
        linecycler = cycle(("-", "--", "-.", ":"))
        print(reward_list)
        for zipped in zip(reward_list, labels, marker, linecycler)
            reward, label, marker_iter, line_iter = zipped
            plot(1:T, reward, label=label,  linestyle=line_iter, linewidth=global_width) 
        end
        ticklabel_format(style="sci", axis="y", scilimits=(0, 0))        
        legend(loc="best")
        xlabel("Iteration index")
        ylabel(L"$(1-1/e)$-Regret")
        grid()
        play_sound()
        my_show()
    end
    function generate_figure_regret_K()
        param_list = 1:20
        function f_cum(x)
            return sum([f(x, params) for params in X_list])
        end
        function gradient_cum(x)
            return sum([gradient(x, params) for params in X_list])
        end
        v = frank_wolfe2(dim_lam, A, b, gradient_cum, K)
        reward_offline = f_cum(v)
        ga1 = online_gradient_ascent(dim_lam, X_list, linear_projection, f, gradient, 0.8, T)[end]
        ga2 = online_gradient_ascent(dim_lam, X_list, linear_projection, f, gradient, 1, T)[end]
        reward_list = zeros(4, size(param_list)[1])
        for (idx, K) in enumerate(param_list)
            idx_th_col = reward_offline - [meta_frank_wolfe(dim_lam, X_list, linear_projection, f, gradient, 0.8, T, K)[end],
                meta_frank_wolfe(dim_lam, X_list, linear_projection, f, gradient, 1, T, K)[end], ga1, ga2]
            reward_list[:, idx] = idx_th_col
        end
        reward_list = [reward_list[i, :] for i in 1:size(reward_list)[1]]
        labels = [L"FW ($\eta=200$)", L"FW ($\eta=100$)", L"GA ($\eta=200$)", L"GA ($\eta=100$)"]        
        marker = cycle((".", ",", "+", "_", "o", "x", "*"))
        linecycler = cycle(("-", "--", "-.", ":"))
        print(reward_list)
        for zipped in zip(reward_list, labels, marker, linecycler)
            reward, label, marker_iter, line_iter = zipped
            plot(param_list, reward, label=label, linestyle=line_iter, linewidth=global_width)
        end
        legend(loc="best")
        ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        xlabel(L"$K$")
        ylabel(L"$(1-1/e)$-Regret")
        grid()
        # savefig("design_regret_K3.pdf")
        play_sound()
        my_show()
    end
    generate_figure_regret(f, gradient)
end
experiment_nqp(10, 100, 100)
# experiment_design(20, 20, 50, 50)
# experiment_coverage(20, 60, 100, 100)