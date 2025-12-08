function [bestCost, tour] = tsp_viterbi_trellis(d, returnToStart, start)
% Viterbi-on-trellis Held–Karp
% - Stage s = #visited nodes (1..n)
% - State at stage s: (mask, last) with |mask|=s, start∈mask, last∈mask
% - Forward: delta{s}(k) = min_{prev in stage s-1} [delta{s-1}(p) + d(prev.last, last)]
% - Backtrace: psi{s}(k) gives predecessor index in stage s-1
%
% Inputs:
%   d (n x n): cost matrix
%   returnToStart (bool): cycle(true) or path(false)
%   start (int): starting node (default 1)
%
% Outputs:
%   bestCost, tour (cycle: length n+1; path: length n)

    if nargin < 2 || isempty(returnToStart), returnToStart = true; end
    if nargin < 3 || isempty(start), start = 1; end

    d = double(d); n = size(d,1);
    if size(d,2) ~= n, error('d must be square'); end
    if start<1 || start>n, error('invalid start'); end

    % bitmasks
    FULL = bitshift(1,n)-1;
    START_MASK = bitset(0,start,1);

    % --------- 1) Trellis 구성: stage별 상태 리스트 ----------
    % stageStates{s} : [mask, last] rows
    stageStates = cell(n,1);

    % stage 1: only {start}, last=start
    stageStates{1} = [START_MASK, start];

    nodes = setdiff(1:n, start);

    % stage 2..n: subset size = s, include start
    for s = 2:n
        combs = nchoosek(nodes, s-1); % choose (s-1) from nodes (excluding start)
        states = [];
        for r = 1:size(combs,1)
            subset = combs(r,:);
            mask = START_MASK;
            for u = subset, mask = bitset(mask, u, 1); end
            % last ∈ subset
            for last = subset
                states = [states; mask, last]; %#ok<AGROW>
            end
        end
        stageStates{s} = states;
    end

    % --------- 2) 인덱스 매핑(빠른 탐색용) ----------
    % key(mask,last) -> index in stage s
    key = @(mask,last)sprintf('%d_%d',mask,last);
    indexMap = cell(n,1);
    for s = 1:n
        states = stageStates{s};
        mp = containers.Map('KeyType','char','ValueType','int32');
        for k = 1:size(states,1)
            mp(key(states(k,1), states(k,2))) = k;
        end
        indexMap{s} = mp;
    end

    % --------- 3) Forward (delta) & Backpointer (psi) ----------
    delta = cell(n,1);
    psi   = cell(n,1);

    delta{1} = inf(size(stageStates{1},1),1);
    psi{1}   = zeros(size(stageStates{1},1),1,'int32');
    % stage 1 cost: start at start with 0
    delta{1}(1) = 0.0;  % only one state

    for s = 2:n
        states = stageStates{s};
        prevStates = stageStates{s-1};
        delta{s} = inf(size(states,1),1);
        psi{s}   = zeros(size(states,1),1,'int32');

        % build quick list of prev states by mask relation
        for k = 1:size(states,1)
            mask = states(k,1); last = states(k,2);
            prevMask = bitset(mask, last, 0);
            % predecessor candidates: (prevMask, prevLast) with prevLast ∈ prevMask
            % iterate prevLast from all nodes in prevMask
            best = inf; argp = int32(0);
            % enumerate prev candidates by scanning prev stage list
            for p = 1:size(prevStates,1)
                if prevStates(p,1) ~= prevMask, continue; end
                prevLast = prevStates(p,2);
                cost = delta{s-1}(p) + d(prevLast, last);
                if cost < best
                    best = cost; argp = int32(p);
                end
            end
            delta{s}(k) = best;
            psi{s}(k)   = argp;
        end
    end

    % --------- 4) Termination ----------
    finalStates = stageStates{n};  % |mask|=n
    if returnToStart
        best = inf; last_idx = int32(0);
        for k = 1:size(finalStates,1)
            last = finalStates(k,2);
            costk = delta{n}(k) + d(last, start);
            if costk < best
                best = costk; last_idx = int32(k);
            end
        end
        bestCost = best;

        % --------- 5) Backtrace (cycle) ----------
        tour = zeros(n+1,1);
        tour(end) = start;
        idx = last_idx;
        for s = n:-1:2
            last = stageStates{s}(idx,2);
            tour(s) = last;
            idx = psi{s}(idx);
        end
        tour(1) = start;
    else
        % path: no return arc
        best = inf; last_idx = int32(0);
        for k = 1:size(finalStates,1)
            if finalStates(k,2) == start && n>1, continue; end % nontrivial
            costk = delta{n}(k);
            if costk < best
                best = costk; last_idx = int32(k);
            end
        end
        bestCost = best;

        % --------- 5) Backtrace (path) ----------
        tour = zeros(n,1);
        idx = last_idx;
        for s = n:-1:2
            last = stageStates{s}(idx,2);
            tour(s) = last;
            idx = psi{s}(idx);
        end
        tour(1) = start;
    end
end
