<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en-US">
<head>
<link rel="icon" href="/cpython/static/hgicon.png" type="image/png" />
<meta name="robots" content="index, nofollow" />
<link rel="stylesheet" href="/cpython/static/style-paper.css" type="text/css" />
<script type="text/javascript" src="/cpython/static/mercurial.js"></script>

<link rel="stylesheet" href="/cpython/highlightcss" type="text/css" />
<title>cpython: 80d4da2e78ec Lib/queue.py</title>
</head>
<body>

<div class="container">
<div class="menu">
<div class="logo">
<a href="https://hg.python.org">
<img src="/cpython/static/hglogo.png" alt="back to hg.python.org repositories" /></a>
</div>
<ul>
<li><a href="/cpython/shortlog/3.4">log</a></li>
<li><a href="/cpython/graph/3.4">graph</a></li>
<li><a href="/cpython/tags">tags</a></li>
<li><a href="/cpython/bookmarks">bookmarks</a></li>
<li><a href="/cpython/branches">branches</a></li>
</ul>
<ul>
<li><a href="/cpython/rev/3.4">changeset</a></li>
<li><a href="/cpython/file/3.4/Lib/">browse</a></li>
</ul>
<ul>
<li class="active">file</li>
<li><a href="/cpython/file/tip/Lib/queue.py">latest</a></li>
<li><a href="/cpython/diff/3.4/Lib/queue.py">diff</a></li>
<li><a href="/cpython/comparison/3.4/Lib/queue.py">comparison</a></li>
<li><a href="/cpython/annotate/3.4/Lib/queue.py">annotate</a></li>
<li><a href="/cpython/log/3.4/Lib/queue.py">file log</a></li>
<li><a href="/cpython/raw-file/3.4/Lib/queue.py">raw</a></li>
</ul>
<ul>
<li><a href="/cpython/help">help</a></li>
</ul>
</div>

<div class="main">
<h2 class="breadcrumb"><a href="/">Mercurial</a> &gt; <a href="/cpython">cpython</a> </h2>
<h3>
 view Lib/queue.py @ 105137:<a href="/cpython/rev/80d4da2e78ec">80d4da2e78ec</a>
 <span class="branchname">3.4</span> 
</h3>

<form class="search" action="/cpython/log">

<p><input name="rev" id="search1" type="text" size="30" /></p>
<div id="hint">Find changesets by keywords (author, files, the commit message), revision
number or hash, or <a href="/cpython/help/revsets">revset expression</a>.</div>
</form>

<div class="description">Upgrade pip to 9.0.1 and setuptools to 28.8.0</div>

<table id="changesetEntry">
<tr>
 <th class="author">author</th>
 <td class="author">&#68;&#111;&#110;&#97;&#108;&#100;&#32;&#83;&#116;&#117;&#102;&#102;&#116;&#32;&#60;&#100;&#111;&#110;&#97;&#108;&#100;&#64;&#115;&#116;&#117;&#102;&#102;&#116;&#46;&#105;&#111;&#62;</td>
</tr>
<tr>
 <th class="date">date</th>
 <td class="date age">Tue, 15 Nov 2016 21:17:43 -0500</td>
</tr>
<tr>
 <th class="author">parents</th>
 <td class="author"><a href="/cpython/file/d9a9fe5e700d/Lib/queue.py">d9a9fe5e700d</a> </td>
</tr>
<tr>
 <th class="author">children</th>
 <td class="author"><a href="/cpython/file/b12857782041/Lib/queue.py">b12857782041</a> </td>
</tr>
</table>

<div class="overflow">
<div class="sourcefirst linewraptoggle">line wrap: <a class="linewraplink" href="javascript:toggleLinewrap()">on</a></div>
<div class="sourcefirst"> line source</div>
<pre class="sourcelines stripes4 wrap">
<span id="l1"><span class="sd">&#39;&#39;&#39;A multi-producer, multi-consumer queue.&#39;&#39;&#39;</span></span><a href="#l1"></a>
<span id="l2"></span><a href="#l2"></a>
<span id="l3"><span class="k">try</span><span class="p">:</span></span><a href="#l3"></a>
<span id="l4">    <span class="kn">import</span> <span class="nn">threading</span></span><a href="#l4"></a>
<span id="l5"><span class="k">except</span> <span class="ne">ImportError</span><span class="p">:</span></span><a href="#l5"></a>
<span id="l6">    <span class="kn">import</span> <span class="nn">dummy_threading</span> <span class="kn">as</span> <span class="nn">threading</span></span><a href="#l6"></a>
<span id="l7"><span class="kn">from</span> <span class="nn">collections</span> <span class="kn">import</span> <span class="n">deque</span></span><a href="#l7"></a>
<span id="l8"><span class="kn">from</span> <span class="nn">heapq</span> <span class="kn">import</span> <span class="n">heappush</span><span class="p">,</span> <span class="n">heappop</span></span><a href="#l8"></a>
<span id="l9"><span class="k">try</span><span class="p">:</span></span><a href="#l9"></a>
<span id="l10">    <span class="kn">from</span> <span class="nn">time</span> <span class="kn">import</span> <span class="n">monotonic</span> <span class="k">as</span> <span class="n">time</span></span><a href="#l10"></a>
<span id="l11"><span class="k">except</span> <span class="ne">ImportError</span><span class="p">:</span></span><a href="#l11"></a>
<span id="l12">    <span class="kn">from</span> <span class="nn">time</span> <span class="kn">import</span> <span class="n">time</span></span><a href="#l12"></a>
<span id="l13"></span><a href="#l13"></a>
<span id="l14"><span class="n">__all__</span> <span class="o">=</span> <span class="p">[</span><span class="s">&#39;Empty&#39;</span><span class="p">,</span> <span class="s">&#39;Full&#39;</span><span class="p">,</span> <span class="s">&#39;Queue&#39;</span><span class="p">,</span> <span class="s">&#39;PriorityQueue&#39;</span><span class="p">,</span> <span class="s">&#39;LifoQueue&#39;</span><span class="p">]</span></span><a href="#l14"></a>
<span id="l15"></span><a href="#l15"></a>
<span id="l16"><span class="k">class</span> <span class="nc">Empty</span><span class="p">(</span><span class="ne">Exception</span><span class="p">):</span></span><a href="#l16"></a>
<span id="l17">    <span class="s">&#39;Exception raised by Queue.get(block=0)/get_nowait().&#39;</span></span><a href="#l17"></a>
<span id="l18">    <span class="k">pass</span></span><a href="#l18"></a>
<span id="l19"></span><a href="#l19"></a>
<span id="l20"><span class="k">class</span> <span class="nc">Full</span><span class="p">(</span><span class="ne">Exception</span><span class="p">):</span></span><a href="#l20"></a>
<span id="l21">    <span class="s">&#39;Exception raised by Queue.put(block=0)/put_nowait().&#39;</span></span><a href="#l21"></a>
<span id="l22">    <span class="k">pass</span></span><a href="#l22"></a>
<span id="l23"></span><a href="#l23"></a>
<span id="l24"><span class="k">class</span> <span class="nc">Queue</span><span class="p">:</span></span><a href="#l24"></a>
<span id="l25">    <span class="sd">&#39;&#39;&#39;Create a queue object with a given maximum size.</span></span><a href="#l25"></a>
<span id="l26"></span><a href="#l26"></a>
<span id="l27"><span class="sd">    If maxsize is &lt;= 0, the queue size is infinite.</span></span><a href="#l27"></a>
<span id="l28"><span class="sd">    &#39;&#39;&#39;</span></span><a href="#l28"></a>
<span id="l29"></span><a href="#l29"></a>
<span id="l30">    <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">maxsize</span><span class="o">=</span><span class="mi">0</span><span class="p">):</span></span><a href="#l30"></a>
<span id="l31">        <span class="bp">self</span><span class="o">.</span><span class="n">maxsize</span> <span class="o">=</span> <span class="n">maxsize</span></span><a href="#l31"></a>
<span id="l32">        <span class="bp">self</span><span class="o">.</span><span class="n">_init</span><span class="p">(</span><span class="n">maxsize</span><span class="p">)</span></span><a href="#l32"></a>
<span id="l33"></span><a href="#l33"></a>
<span id="l34">        <span class="c"># mutex must be held whenever the queue is mutating.  All methods</span></span><a href="#l34"></a>
<span id="l35">        <span class="c"># that acquire mutex must release it before returning.  mutex</span></span><a href="#l35"></a>
<span id="l36">        <span class="c"># is shared between the three conditions, so acquiring and</span></span><a href="#l36"></a>
<span id="l37">        <span class="c"># releasing the conditions also acquires and releases mutex.</span></span><a href="#l37"></a>
<span id="l38">        <span class="bp">self</span><span class="o">.</span><span class="n">mutex</span> <span class="o">=</span> <span class="n">threading</span><span class="o">.</span><span class="n">Lock</span><span class="p">()</span></span><a href="#l38"></a>
<span id="l39"></span><a href="#l39"></a>
<span id="l40">        <span class="c"># Notify not_empty whenever an item is added to the queue; a</span></span><a href="#l40"></a>
<span id="l41">        <span class="c"># thread waiting to get is notified then.</span></span><a href="#l41"></a>
<span id="l42">        <span class="bp">self</span><span class="o">.</span><span class="n">not_empty</span> <span class="o">=</span> <span class="n">threading</span><span class="o">.</span><span class="n">Condition</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">mutex</span><span class="p">)</span></span><a href="#l42"></a>
<span id="l43"></span><a href="#l43"></a>
<span id="l44">        <span class="c"># Notify not_full whenever an item is removed from the queue;</span></span><a href="#l44"></a>
<span id="l45">        <span class="c"># a thread waiting to put is notified then.</span></span><a href="#l45"></a>
<span id="l46">        <span class="bp">self</span><span class="o">.</span><span class="n">not_full</span> <span class="o">=</span> <span class="n">threading</span><span class="o">.</span><span class="n">Condition</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">mutex</span><span class="p">)</span></span><a href="#l46"></a>
<span id="l47"></span><a href="#l47"></a>
<span id="l48">        <span class="c"># Notify all_tasks_done whenever the number of unfinished tasks</span></span><a href="#l48"></a>
<span id="l49">        <span class="c"># drops to zero; thread waiting to join() is notified to resume</span></span><a href="#l49"></a>
<span id="l50">        <span class="bp">self</span><span class="o">.</span><span class="n">all_tasks_done</span> <span class="o">=</span> <span class="n">threading</span><span class="o">.</span><span class="n">Condition</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">mutex</span><span class="p">)</span></span><a href="#l50"></a>
<span id="l51">        <span class="bp">self</span><span class="o">.</span><span class="n">unfinished_tasks</span> <span class="o">=</span> <span class="mi">0</span></span><a href="#l51"></a>
<span id="l52"></span><a href="#l52"></a>
<span id="l53">    <span class="k">def</span> <span class="nf">task_done</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span></span><a href="#l53"></a>
<span id="l54">        <span class="sd">&#39;&#39;&#39;Indicate that a formerly enqueued task is complete.</span></span><a href="#l54"></a>
<span id="l55"></span><a href="#l55"></a>
<span id="l56"><span class="sd">        Used by Queue consumer threads.  For each get() used to fetch a task,</span></span><a href="#l56"></a>
<span id="l57"><span class="sd">        a subsequent call to task_done() tells the queue that the processing</span></span><a href="#l57"></a>
<span id="l58"><span class="sd">        on the task is complete.</span></span><a href="#l58"></a>
<span id="l59"></span><a href="#l59"></a>
<span id="l60"><span class="sd">        If a join() is currently blocking, it will resume when all items</span></span><a href="#l60"></a>
<span id="l61"><span class="sd">        have been processed (meaning that a task_done() call was received</span></span><a href="#l61"></a>
<span id="l62"><span class="sd">        for every item that had been put() into the queue).</span></span><a href="#l62"></a>
<span id="l63"></span><a href="#l63"></a>
<span id="l64"><span class="sd">        Raises a ValueError if called more times than there were items</span></span><a href="#l64"></a>
<span id="l65"><span class="sd">        placed in the queue.</span></span><a href="#l65"></a>
<span id="l66"><span class="sd">        &#39;&#39;&#39;</span></span><a href="#l66"></a>
<span id="l67">        <span class="k">with</span> <span class="bp">self</span><span class="o">.</span><span class="n">all_tasks_done</span><span class="p">:</span></span><a href="#l67"></a>
<span id="l68">            <span class="n">unfinished</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">unfinished_tasks</span> <span class="o">-</span> <span class="mi">1</span></span><a href="#l68"></a>
<span id="l69">            <span class="k">if</span> <span class="n">unfinished</span> <span class="o">&lt;=</span> <span class="mi">0</span><span class="p">:</span></span><a href="#l69"></a>
<span id="l70">                <span class="k">if</span> <span class="n">unfinished</span> <span class="o">&lt;</span> <span class="mi">0</span><span class="p">:</span></span><a href="#l70"></a>
<span id="l71">                    <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span><span class="s">&#39;task_done() called too many times&#39;</span><span class="p">)</span></span><a href="#l71"></a>
<span id="l72">                <span class="bp">self</span><span class="o">.</span><span class="n">all_tasks_done</span><span class="o">.</span><span class="n">notify_all</span><span class="p">()</span></span><a href="#l72"></a>
<span id="l73">            <span class="bp">self</span><span class="o">.</span><span class="n">unfinished_tasks</span> <span class="o">=</span> <span class="n">unfinished</span></span><a href="#l73"></a>
<span id="l74"></span><a href="#l74"></a>
<span id="l75">    <span class="k">def</span> <span class="nf">join</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span></span><a href="#l75"></a>
<span id="l76">        <span class="sd">&#39;&#39;&#39;Blocks until all items in the Queue have been gotten and processed.</span></span><a href="#l76"></a>
<span id="l77"></span><a href="#l77"></a>
<span id="l78"><span class="sd">        The count of unfinished tasks goes up whenever an item is added to the</span></span><a href="#l78"></a>
<span id="l79"><span class="sd">        queue. The count goes down whenever a consumer thread calls task_done()</span></span><a href="#l79"></a>
<span id="l80"><span class="sd">        to indicate the item was retrieved and all work on it is complete.</span></span><a href="#l80"></a>
<span id="l81"></span><a href="#l81"></a>
<span id="l82"><span class="sd">        When the count of unfinished tasks drops to zero, join() unblocks.</span></span><a href="#l82"></a>
<span id="l83"><span class="sd">        &#39;&#39;&#39;</span></span><a href="#l83"></a>
<span id="l84">        <span class="k">with</span> <span class="bp">self</span><span class="o">.</span><span class="n">all_tasks_done</span><span class="p">:</span></span><a href="#l84"></a>
<span id="l85">            <span class="k">while</span> <span class="bp">self</span><span class="o">.</span><span class="n">unfinished_tasks</span><span class="p">:</span></span><a href="#l85"></a>
<span id="l86">                <span class="bp">self</span><span class="o">.</span><span class="n">all_tasks_done</span><span class="o">.</span><span class="n">wait</span><span class="p">()</span></span><a href="#l86"></a>
<span id="l87"></span><a href="#l87"></a>
<span id="l88">    <span class="k">def</span> <span class="nf">qsize</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span></span><a href="#l88"></a>
<span id="l89">        <span class="sd">&#39;&#39;&#39;Return the approximate size of the queue (not reliable!).&#39;&#39;&#39;</span></span><a href="#l89"></a>
<span id="l90">        <span class="k">with</span> <span class="bp">self</span><span class="o">.</span><span class="n">mutex</span><span class="p">:</span></span><a href="#l90"></a>
<span id="l91">            <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">_qsize</span><span class="p">()</span></span><a href="#l91"></a>
<span id="l92"></span><a href="#l92"></a>
<span id="l93">    <span class="k">def</span> <span class="nf">empty</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span></span><a href="#l93"></a>
<span id="l94">        <span class="sd">&#39;&#39;&#39;Return True if the queue is empty, False otherwise (not reliable!).</span></span><a href="#l94"></a>
<span id="l95"></span><a href="#l95"></a>
<span id="l96"><span class="sd">        This method is likely to be removed at some point.  Use qsize() == 0</span></span><a href="#l96"></a>
<span id="l97"><span class="sd">        as a direct substitute, but be aware that either approach risks a race</span></span><a href="#l97"></a>
<span id="l98"><span class="sd">        condition where a queue can grow before the result of empty() or</span></span><a href="#l98"></a>
<span id="l99"><span class="sd">        qsize() can be used.</span></span><a href="#l99"></a>
<span id="l100"></span><a href="#l100"></a>
<span id="l101"><span class="sd">        To create code that needs to wait for all queued tasks to be</span></span><a href="#l101"></a>
<span id="l102"><span class="sd">        completed, the preferred technique is to use the join() method.</span></span><a href="#l102"></a>
<span id="l103"><span class="sd">        &#39;&#39;&#39;</span></span><a href="#l103"></a>
<span id="l104">        <span class="k">with</span> <span class="bp">self</span><span class="o">.</span><span class="n">mutex</span><span class="p">:</span></span><a href="#l104"></a>
<span id="l105">            <span class="k">return</span> <span class="ow">not</span> <span class="bp">self</span><span class="o">.</span><span class="n">_qsize</span><span class="p">()</span></span><a href="#l105"></a>
<span id="l106"></span><a href="#l106"></a>
<span id="l107">    <span class="k">def</span> <span class="nf">full</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span></span><a href="#l107"></a>
<span id="l108">        <span class="sd">&#39;&#39;&#39;Return True if the queue is full, False otherwise (not reliable!).</span></span><a href="#l108"></a>
<span id="l109"></span><a href="#l109"></a>
<span id="l110"><span class="sd">        This method is likely to be removed at some point.  Use qsize() &gt;= n</span></span><a href="#l110"></a>
<span id="l111"><span class="sd">        as a direct substitute, but be aware that either approach risks a race</span></span><a href="#l111"></a>
<span id="l112"><span class="sd">        condition where a queue can shrink before the result of full() or</span></span><a href="#l112"></a>
<span id="l113"><span class="sd">        qsize() can be used.</span></span><a href="#l113"></a>
<span id="l114"><span class="sd">        &#39;&#39;&#39;</span></span><a href="#l114"></a>
<span id="l115">        <span class="k">with</span> <span class="bp">self</span><span class="o">.</span><span class="n">mutex</span><span class="p">:</span></span><a href="#l115"></a>
<span id="l116">            <span class="k">return</span> <span class="mi">0</span> <span class="o">&lt;</span> <span class="bp">self</span><span class="o">.</span><span class="n">maxsize</span> <span class="o">&lt;=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_qsize</span><span class="p">()</span></span><a href="#l116"></a>
<span id="l117"></span><a href="#l117"></a>
<span id="l118">    <span class="k">def</span> <span class="nf">put</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">item</span><span class="p">,</span> <span class="n">block</span><span class="o">=</span><span class="bp">True</span><span class="p">,</span> <span class="n">timeout</span><span class="o">=</span><span class="bp">None</span><span class="p">):</span></span><a href="#l118"></a>
<span id="l119">        <span class="sd">&#39;&#39;&#39;Put an item into the queue.</span></span><a href="#l119"></a>
<span id="l120"></span><a href="#l120"></a>
<span id="l121"><span class="sd">        If optional args &#39;block&#39; is true and &#39;timeout&#39; is None (the default),</span></span><a href="#l121"></a>
<span id="l122"><span class="sd">        block if necessary until a free slot is available. If &#39;timeout&#39; is</span></span><a href="#l122"></a>
<span id="l123"><span class="sd">        a non-negative number, it blocks at most &#39;timeout&#39; seconds and raises</span></span><a href="#l123"></a>
<span id="l124"><span class="sd">        the Full exception if no free slot was available within that time.</span></span><a href="#l124"></a>
<span id="l125"><span class="sd">        Otherwise (&#39;block&#39; is false), put an item on the queue if a free slot</span></span><a href="#l125"></a>
<span id="l126"><span class="sd">        is immediately available, else raise the Full exception (&#39;timeout&#39;</span></span><a href="#l126"></a>
<span id="l127"><span class="sd">        is ignored in that case).</span></span><a href="#l127"></a>
<span id="l128"><span class="sd">        &#39;&#39;&#39;</span></span><a href="#l128"></a>
<span id="l129">        <span class="k">with</span> <span class="bp">self</span><span class="o">.</span><span class="n">not_full</span><span class="p">:</span></span><a href="#l129"></a>
<span id="l130">            <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">maxsize</span> <span class="o">&gt;</span> <span class="mi">0</span><span class="p">:</span></span><a href="#l130"></a>
<span id="l131">                <span class="k">if</span> <span class="ow">not</span> <span class="n">block</span><span class="p">:</span></span><a href="#l131"></a>
<span id="l132">                    <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">_qsize</span><span class="p">()</span> <span class="o">&gt;=</span> <span class="bp">self</span><span class="o">.</span><span class="n">maxsize</span><span class="p">:</span></span><a href="#l132"></a>
<span id="l133">                        <span class="k">raise</span> <span class="n">Full</span></span><a href="#l133"></a>
<span id="l134">                <span class="k">elif</span> <span class="n">timeout</span> <span class="ow">is</span> <span class="bp">None</span><span class="p">:</span></span><a href="#l134"></a>
<span id="l135">                    <span class="k">while</span> <span class="bp">self</span><span class="o">.</span><span class="n">_qsize</span><span class="p">()</span> <span class="o">&gt;=</span> <span class="bp">self</span><span class="o">.</span><span class="n">maxsize</span><span class="p">:</span></span><a href="#l135"></a>
<span id="l136">                        <span class="bp">self</span><span class="o">.</span><span class="n">not_full</span><span class="o">.</span><span class="n">wait</span><span class="p">()</span></span><a href="#l136"></a>
<span id="l137">                <span class="k">elif</span> <span class="n">timeout</span> <span class="o">&lt;</span> <span class="mi">0</span><span class="p">:</span></span><a href="#l137"></a>
<span id="l138">                    <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span><span class="s">&quot;&#39;timeout&#39; must be a non-negative number&quot;</span><span class="p">)</span></span><a href="#l138"></a>
<span id="l139">                <span class="k">else</span><span class="p">:</span></span><a href="#l139"></a>
<span id="l140">                    <span class="n">endtime</span> <span class="o">=</span> <span class="n">time</span><span class="p">()</span> <span class="o">+</span> <span class="n">timeout</span></span><a href="#l140"></a>
<span id="l141">                    <span class="k">while</span> <span class="bp">self</span><span class="o">.</span><span class="n">_qsize</span><span class="p">()</span> <span class="o">&gt;=</span> <span class="bp">self</span><span class="o">.</span><span class="n">maxsize</span><span class="p">:</span></span><a href="#l141"></a>
<span id="l142">                        <span class="n">remaining</span> <span class="o">=</span> <span class="n">endtime</span> <span class="o">-</span> <span class="n">time</span><span class="p">()</span></span><a href="#l142"></a>
<span id="l143">                        <span class="k">if</span> <span class="n">remaining</span> <span class="o">&lt;=</span> <span class="mf">0.0</span><span class="p">:</span></span><a href="#l143"></a>
<span id="l144">                            <span class="k">raise</span> <span class="n">Full</span></span><a href="#l144"></a>
<span id="l145">                        <span class="bp">self</span><span class="o">.</span><span class="n">not_full</span><span class="o">.</span><span class="n">wait</span><span class="p">(</span><span class="n">remaining</span><span class="p">)</span></span><a href="#l145"></a>
<span id="l146">            <span class="bp">self</span><span class="o">.</span><span class="n">_put</span><span class="p">(</span><span class="n">item</span><span class="p">)</span></span><a href="#l146"></a>
<span id="l147">            <span class="bp">self</span><span class="o">.</span><span class="n">unfinished_tasks</span> <span class="o">+=</span> <span class="mi">1</span></span><a href="#l147"></a>
<span id="l148">            <span class="bp">self</span><span class="o">.</span><span class="n">not_empty</span><span class="o">.</span><span class="n">notify</span><span class="p">()</span></span><a href="#l148"></a>
<span id="l149"></span><a href="#l149"></a>
<span id="l150">    <span class="k">def</span> <span class="nf">get</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">block</span><span class="o">=</span><span class="bp">True</span><span class="p">,</span> <span class="n">timeout</span><span class="o">=</span><span class="bp">None</span><span class="p">):</span></span><a href="#l150"></a>
<span id="l151">        <span class="sd">&#39;&#39;&#39;Remove and return an item from the queue.</span></span><a href="#l151"></a>
<span id="l152"></span><a href="#l152"></a>
<span id="l153"><span class="sd">        If optional args &#39;block&#39; is true and &#39;timeout&#39; is None (the default),</span></span><a href="#l153"></a>
<span id="l154"><span class="sd">        block if necessary until an item is available. If &#39;timeout&#39; is</span></span><a href="#l154"></a>
<span id="l155"><span class="sd">        a non-negative number, it blocks at most &#39;timeout&#39; seconds and raises</span></span><a href="#l155"></a>
<span id="l156"><span class="sd">        the Empty exception if no item was available within that time.</span></span><a href="#l156"></a>
<span id="l157"><span class="sd">        Otherwise (&#39;block&#39; is false), return an item if one is immediately</span></span><a href="#l157"></a>
<span id="l158"><span class="sd">        available, else raise the Empty exception (&#39;timeout&#39; is ignored</span></span><a href="#l158"></a>
<span id="l159"><span class="sd">        in that case).</span></span><a href="#l159"></a>
<span id="l160"><span class="sd">        &#39;&#39;&#39;</span></span><a href="#l160"></a>
<span id="l161">        <span class="k">with</span> <span class="bp">self</span><span class="o">.</span><span class="n">not_empty</span><span class="p">:</span></span><a href="#l161"></a>
<span id="l162">            <span class="k">if</span> <span class="ow">not</span> <span class="n">block</span><span class="p">:</span></span><a href="#l162"></a>
<span id="l163">                <span class="k">if</span> <span class="ow">not</span> <span class="bp">self</span><span class="o">.</span><span class="n">_qsize</span><span class="p">():</span></span><a href="#l163"></a>
<span id="l164">                    <span class="k">raise</span> <span class="n">Empty</span></span><a href="#l164"></a>
<span id="l165">            <span class="k">elif</span> <span class="n">timeout</span> <span class="ow">is</span> <span class="bp">None</span><span class="p">:</span></span><a href="#l165"></a>
<span id="l166">                <span class="k">while</span> <span class="ow">not</span> <span class="bp">self</span><span class="o">.</span><span class="n">_qsize</span><span class="p">():</span></span><a href="#l166"></a>
<span id="l167">                    <span class="bp">self</span><span class="o">.</span><span class="n">not_empty</span><span class="o">.</span><span class="n">wait</span><span class="p">()</span></span><a href="#l167"></a>
<span id="l168">            <span class="k">elif</span> <span class="n">timeout</span> <span class="o">&lt;</span> <span class="mi">0</span><span class="p">:</span></span><a href="#l168"></a>
<span id="l169">                <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span><span class="s">&quot;&#39;timeout&#39; must be a non-negative number&quot;</span><span class="p">)</span></span><a href="#l169"></a>
<span id="l170">            <span class="k">else</span><span class="p">:</span></span><a href="#l170"></a>
<span id="l171">                <span class="n">endtime</span> <span class="o">=</span> <span class="n">time</span><span class="p">()</span> <span class="o">+</span> <span class="n">timeout</span></span><a href="#l171"></a>
<span id="l172">                <span class="k">while</span> <span class="ow">not</span> <span class="bp">self</span><span class="o">.</span><span class="n">_qsize</span><span class="p">():</span></span><a href="#l172"></a>
<span id="l173">                    <span class="n">remaining</span> <span class="o">=</span> <span class="n">endtime</span> <span class="o">-</span> <span class="n">time</span><span class="p">()</span></span><a href="#l173"></a>
<span id="l174">                    <span class="k">if</span> <span class="n">remaining</span> <span class="o">&lt;=</span> <span class="mf">0.0</span><span class="p">:</span></span><a href="#l174"></a>
<span id="l175">                        <span class="k">raise</span> <span class="n">Empty</span></span><a href="#l175"></a>
<span id="l176">                    <span class="bp">self</span><span class="o">.</span><span class="n">not_empty</span><span class="o">.</span><span class="n">wait</span><span class="p">(</span><span class="n">remaining</span><span class="p">)</span></span><a href="#l176"></a>
<span id="l177">            <span class="n">item</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_get</span><span class="p">()</span></span><a href="#l177"></a>
<span id="l178">            <span class="bp">self</span><span class="o">.</span><span class="n">not_full</span><span class="o">.</span><span class="n">notify</span><span class="p">()</span></span><a href="#l178"></a>
<span id="l179">            <span class="k">return</span> <span class="n">item</span></span><a href="#l179"></a>
<span id="l180"></span><a href="#l180"></a>
<span id="l181">    <span class="k">def</span> <span class="nf">put_nowait</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">item</span><span class="p">):</span></span><a href="#l181"></a>
<span id="l182">        <span class="sd">&#39;&#39;&#39;Put an item into the queue without blocking.</span></span><a href="#l182"></a>
<span id="l183"></span><a href="#l183"></a>
<span id="l184"><span class="sd">        Only enqueue the item if a free slot is immediately available.</span></span><a href="#l184"></a>
<span id="l185"><span class="sd">        Otherwise raise the Full exception.</span></span><a href="#l185"></a>
<span id="l186"><span class="sd">        &#39;&#39;&#39;</span></span><a href="#l186"></a>
<span id="l187">        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">put</span><span class="p">(</span><span class="n">item</span><span class="p">,</span> <span class="n">block</span><span class="o">=</span><span class="bp">False</span><span class="p">)</span></span><a href="#l187"></a>
<span id="l188"></span><a href="#l188"></a>
<span id="l189">    <span class="k">def</span> <span class="nf">get_nowait</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span></span><a href="#l189"></a>
<span id="l190">        <span class="sd">&#39;&#39;&#39;Remove and return an item from the queue without blocking.</span></span><a href="#l190"></a>
<span id="l191"></span><a href="#l191"></a>
<span id="l192"><span class="sd">        Only get an item if one is immediately available. Otherwise</span></span><a href="#l192"></a>
<span id="l193"><span class="sd">        raise the Empty exception.</span></span><a href="#l193"></a>
<span id="l194"><span class="sd">        &#39;&#39;&#39;</span></span><a href="#l194"></a>
<span id="l195">        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="n">block</span><span class="o">=</span><span class="bp">False</span><span class="p">)</span></span><a href="#l195"></a>
<span id="l196"></span><a href="#l196"></a>
<span id="l197">    <span class="c"># Override these methods to implement other queue organizations</span></span><a href="#l197"></a>
<span id="l198">    <span class="c"># (e.g. stack or priority queue).</span></span><a href="#l198"></a>
<span id="l199">    <span class="c"># These will only be called with appropriate locks held</span></span><a href="#l199"></a>
<span id="l200"></span><a href="#l200"></a>
<span id="l201">    <span class="c"># Initialize the queue representation</span></span><a href="#l201"></a>
<span id="l202">    <span class="k">def</span> <span class="nf">_init</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">maxsize</span><span class="p">):</span></span><a href="#l202"></a>
<span id="l203">        <span class="bp">self</span><span class="o">.</span><span class="n">queue</span> <span class="o">=</span> <span class="n">deque</span><span class="p">()</span></span><a href="#l203"></a>
<span id="l204"></span><a href="#l204"></a>
<span id="l205">    <span class="k">def</span> <span class="nf">_qsize</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span></span><a href="#l205"></a>
<span id="l206">        <span class="k">return</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">queue</span><span class="p">)</span></span><a href="#l206"></a>
<span id="l207"></span><a href="#l207"></a>
<span id="l208">    <span class="c"># Put a new item in the queue</span></span><a href="#l208"></a>
<span id="l209">    <span class="k">def</span> <span class="nf">_put</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">item</span><span class="p">):</span></span><a href="#l209"></a>
<span id="l210">        <span class="bp">self</span><span class="o">.</span><span class="n">queue</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">item</span><span class="p">)</span></span><a href="#l210"></a>
<span id="l211"></span><a href="#l211"></a>
<span id="l212">    <span class="c"># Get an item from the queue</span></span><a href="#l212"></a>
<span id="l213">    <span class="k">def</span> <span class="nf">_get</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span></span><a href="#l213"></a>
<span id="l214">        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">queue</span><span class="o">.</span><span class="n">popleft</span><span class="p">()</span></span><a href="#l214"></a>
<span id="l215"></span><a href="#l215"></a>
<span id="l216"></span><a href="#l216"></a>
<span id="l217"><span class="k">class</span> <span class="nc">PriorityQueue</span><span class="p">(</span><span class="n">Queue</span><span class="p">):</span></span><a href="#l217"></a>
<span id="l218">    <span class="sd">&#39;&#39;&#39;Variant of Queue that retrieves open entries in priority order (lowest first).</span></span><a href="#l218"></a>
<span id="l219"></span><a href="#l219"></a>
<span id="l220"><span class="sd">    Entries are typically tuples of the form:  (priority number, data).</span></span><a href="#l220"></a>
<span id="l221"><span class="sd">    &#39;&#39;&#39;</span></span><a href="#l221"></a>
<span id="l222"></span><a href="#l222"></a>
<span id="l223">    <span class="k">def</span> <span class="nf">_init</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">maxsize</span><span class="p">):</span></span><a href="#l223"></a>
<span id="l224">        <span class="bp">self</span><span class="o">.</span><span class="n">queue</span> <span class="o">=</span> <span class="p">[]</span></span><a href="#l224"></a>
<span id="l225"></span><a href="#l225"></a>
<span id="l226">    <span class="k">def</span> <span class="nf">_qsize</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span></span><a href="#l226"></a>
<span id="l227">        <span class="k">return</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">queue</span><span class="p">)</span></span><a href="#l227"></a>
<span id="l228"></span><a href="#l228"></a>
<span id="l229">    <span class="k">def</span> <span class="nf">_put</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">item</span><span class="p">):</span></span><a href="#l229"></a>
<span id="l230">        <span class="n">heappush</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">queue</span><span class="p">,</span> <span class="n">item</span><span class="p">)</span></span><a href="#l230"></a>
<span id="l231"></span><a href="#l231"></a>
<span id="l232">    <span class="k">def</span> <span class="nf">_get</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span></span><a href="#l232"></a>
<span id="l233">        <span class="k">return</span> <span class="n">heappop</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">queue</span><span class="p">)</span></span><a href="#l233"></a>
<span id="l234"></span><a href="#l234"></a>
<span id="l235"></span><a href="#l235"></a>
<span id="l236"><span class="k">class</span> <span class="nc">LifoQueue</span><span class="p">(</span><span class="n">Queue</span><span class="p">):</span></span><a href="#l236"></a>
<span id="l237">    <span class="sd">&#39;&#39;&#39;Variant of Queue that retrieves most recently added entries first.&#39;&#39;&#39;</span></span><a href="#l237"></a>
<span id="l238"></span><a href="#l238"></a>
<span id="l239">    <span class="k">def</span> <span class="nf">_init</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">maxsize</span><span class="p">):</span></span><a href="#l239"></a>
<span id="l240">        <span class="bp">self</span><span class="o">.</span><span class="n">queue</span> <span class="o">=</span> <span class="p">[]</span></span><a href="#l240"></a>
<span id="l241"></span><a href="#l241"></a>
<span id="l242">    <span class="k">def</span> <span class="nf">_qsize</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span></span><a href="#l242"></a>
<span id="l243">        <span class="k">return</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">queue</span><span class="p">)</span></span><a href="#l243"></a>
<span id="l244"></span><a href="#l244"></a>
<span id="l245">    <span class="k">def</span> <span class="nf">_put</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">item</span><span class="p">):</span></span><a href="#l245"></a>
<span id="l246">        <span class="bp">self</span><span class="o">.</span><span class="n">queue</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">item</span><span class="p">)</span></span><a href="#l246"></a>
<span id="l247"></span><a href="#l247"></a>
<span id="l248">    <span class="k">def</span> <span class="nf">_get</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span></span><a href="#l248"></a>
<span id="l249">        <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">queue</span><span class="o">.</span><span class="n">pop</span><span class="p">()</span></span><a href="#l249"></a></pre>
<div class="sourcelast"></div>
</div>
</div>
</div>

<script type="text/javascript">process_dates()</script>


</body>
</html>

