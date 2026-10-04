---
layout: post
title: "Minerva RAG System: Implementation History"
---

Minerva is a local RAG system for my personal notes. It runs on my Mac and AI agents like Claude Desktop use it through an MCP server. This post tells how I built it twice: first by letting an AI agent decide almost everything, then again from scratch, measuring every choice. You will find failures (many), a few victories and what I learned about building RAG with small local models.

## First Legacy Minerva

It was autumn 2025 and, after reading and studying, I wanted to explore the world of AI-based applications.
There were many AI-related tools that I could build but my crux has always been finding information: I stuff information in my notes and then I always struggle to find it, so a [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) system seemed the perfect tool for me.

I had no idea how to build RAG systems, I had only read some articles here and there.
In order to see one at work I just told my AI agent to build one. I used Claude Code with a [cool tool built by a colleague, named dev-workflow](https://github.com/andreaserradev-gbj/dev-workflow). It took me one month.

The AI agent decided most of the details while I picked the technology (Python), the database type (Chroma, no servers or remote dbs, plain files) and how the application should behave (two steps: extractors transform data from the native source to a common intermediate markdown + json format).

What did it do? Embed and store markdown notes after having chunked them and provide a dense vector search thanks to Chroma vector [HNSW](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) search. Embeddings were obtained from Ollama at first, later also LM Studio, OpenAI and Gemini.

I built an [MCP server](https://modelcontextprotocol.io/); at the beginning, I tried to use it with a custom implemented chat but I wasn't satisfied so I went back to MCP only, using it with AI desktop applications like Claude Desktop; it always found what I was looking for and I was satisfied (when I analytically evaluated it, I discovered it wasn't _that_ good, read below!).

I published Minerva in December 2025 [on GitHub](https://github.com/michele-orsini-71/minerva).

## Minerva2 project starts

After a few months (it was March 2026) I already had more knowledge about RAGs: I had read [Contextual Retrieval in AI Systems](https://www.anthropic.com/engineering/contextual-retrieval) and Chapter 6 "RAG and Agents" in the book AI Engineering (O'Reilly, 2025) by Chip Huyen (read it guys, it is enlightening). How little I knew about RAGs when I built Minerva!! It was time for a Minerva on steroids.

What does "contextualization" mean? An LLM writes a short context for each chunk you are embedding/indexing. This context is based on the whole document and improves the retrieval every time the chunk does not explain clearly what it is referring to.

This was the whole process

- add contextualization during ingestion, contextualization will help both the full text and the semantic search
- search the query with keyword search
- search the query with embeddings
- RRF ([Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)) merges the ranked lists of the two search methods. In Minerva I used a weighted RRF: `score(chunk) = α/(k + rank_vector) + (1 − α)/(k + rank_fts)`. Where: `k` is the RRF constant. It is fixed at 60. `α` (alpha, HybridAlpha) is the weight between the two lists: `α` = 1 means pure vector search, `α` = 0 means pure full-text search, and `α` = 0.5 gives both equal weight. rank_vector and rank_fts are the chunk's positions in each list.
- add a reranker, later

The Anthropic article claimed: 35% fewer retrieval failures with contextual embeddings, and 67% fewer with [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) and a reranker added, it looked promising.

I had some clear ideas in mind

- a local system: I wanted to build a personal knowledge search system, able to run offline both when indexing and when searching; this was a little different from legacy Minerva, which was also able to use remote systems
- no more two steps, bad UX, I wanted a single operation, without intermediate storage, from the original data to the embeddings in the database
- json configuration files of legacy Minerva were a mess and the whole system was too complex to configure to the point that I needed to add two helpers `minerva-doc` and `minerva-kb` to simplify the user's (that is: my) life
- no more Python, I wanted non-dev people to be able to download a package and install it without having to deal with git, pip/pipx and the terminal if possible
- what about having a UI app to manage everything? Cool!

For this reason, I decided to use C#/.NET. I also thought that coding would be simpler:

- readability
- strong typing
- has lots of libraries and frameworks, including the official MCP/AI SDKs, to speed up development

Then, still with Chroma? Nooo, I wanted to use a full-fledged DB that could provide me both vector and keyword search, so I went with PostgreSQL + pgvector + built-in fts search (please note, not BM25 as in the Anthropic article): both "legs" of retrieval, lexical (full-text) + dense vectors in one shot. Was it a wise choice? More on this later.
Actually, a C# preliminary feasibility study recommended Qdrant but I rejected it because it needs Docker or the cloud. PostgreSQL wasn't an easy install either but it seemed simple. Also, I wanted to learn how to use it so most probably I lost my focus a little here.

At this point, I was ready to implement Anthropic's recipe: dense + full-text, fused with RRF, enhanced by contextualization.

## A first clash with reality

Around April-May 2026 I had my first implementation:

- a Core library
- an indexer CLI tool
- a CLI search command to verify query results

Cool, but some of my previous decisions came back to bite me.

This was the easier move: I had to ditch Microsoft's Host and DI because it was preventing me from building a solid preflight system to check for all the possible prerequisites before starting indexing and eventually fail with a meaningful error.
The problem here was that Microsoft's framework (that's expected) decided when to create objects and object creation failed with wrong settings, so I had to check the settings by myself and then create objects explicitly.

The real pain came from my poor knowledge of local model limitations: Want to build a local system? Do you have the slightest idea of how poor local model performance is? Are you aware that your poor MacBook will suffer with all those inference calls? Ok, let's start

First, contextualization was not done as Anthropic described because I cannot send a whole document to a small model and expect it to be able to ingest it. So I had to split big docs into fragments. Then build a summary of the fragments and finally ask for a contextualization of the chunk using this summary. Not exactly what was recommended by the article, I didn't know it yet, but this compromise would come back.

Then, do you have any idea of how much time ingestion requires when asking for contextualization? 99% of the time was LLM calls for it.
Here are the ingestion timings on ~1.8k docs (my personal notes, taken as a first sample for indexing):

- FTS-only (full-text search) ~10 min
- contextualization using a very small model (Qwen2.5-0.5B) 10.7 h
- contextualization using a small model (Gemma 4 E4B) 52.4 h

The problem was clear: each doc was split into chunks, chunks were around ten thousand, and every chunk had its LLM call for contextualization and this required 1-20s, depending on the model used.

This put the idea of a local system at risk! How could I proceed? Then, well, _measuring_: didn't that lovely book about AI Engineering always talk about building evaluation pipelines? Let's do it.

## OMG, building an eval pipeline is so hard!

The long path towards having my eval pipeline started in May and lasted 4 months (holidays and interruptions due to frustrations included).

Ok, what are the ingredients for an eval pipeline?

1. a corpus to ingest: a set of documents to index and to query against
2. a set of annotated queries, that is: the query to the RAG system and the "gold source", that is, the right document to return (not the right chunk? well, see below)
3. an eval runner, the test bench: a system that automatically runs all the queries, possibly with varying search params and stores results on disk
4. a run analyzer that looks at the results and extracts insights

I first decided the shape of the queries and wrote the bench to run the sweeps, an eval runner that was able to vary search params (topk, alpha blending of hybrid search and, later, reranker on/off switch) using a "sweep" toml file, very cool.

At that point, what I needed was a dataset and the queries and these, of course, must be built in conjunction.

First I had to pick a corpus. Previously I was using my own notes but I wanted something that someone, downloading my repository, should be able to experience themselves, so I used Wikipedia articles, a versioned ZIM file containing a snapshot of a Wikipedia article collection publicly available on Kiwix servers.

With all the moving parts around, I decided to use articles, not chunks, as gold sources; otherwise I would have been forced to redefine gold sources as soon as the corpus or chunking changed. It was a shortcut, actually: the RAG returns the best matching chunk and I only verify that this belonged to the source document I used as gold.

Always with compute time in mind, I started

- 100 docs, some queries
- 200 docs, more queries
- ...
- lots of docs and queries

First run of the bench: all golden sources found. That's too easy: how could I make it more difficult? If I wanted to see improvements in future development, I needed to build more-difficult-to-find queries.

I found that queries were too easy to find, so I crafted them so they did not reference words in title or in the document corpus but expressed the same concepts with different words. It was not easy because several times I went too far, for example linking concepts too loosely: only a "hopping" RAG system would have caught them (but this was not in the scope of Minerva, see below) and I had to rewrite or remove them.

Later, but it is worth mentioning it here where we are talking about corpus and queries, I also added queries that mention acronyms or codes, things that a semantic engine may not store and that could be caught by the other leg of hybrid search (that's the reason why hybrid search has a full-text leg).

To read the bench results, I built the lovely Jupyter notebook I'm still using now. It is called "runs_comparer" and is able (surprised?) to compare runs on the evals showing their differences in

- [Recall](https://en.wikipedia.org/wiki/Precision_and_recall) (R@K for example R@10, R@20 means the recall value for the first 10 or 20 results returned by the RAG)
- [Mean Reciprocal Rank (MRR)](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
- Covering rank, that is the number of results a client has to read to find all the gold sources

It also shows the resulting ranks for each query for each of the sweep params and the gold sources that are "late" (highest rank). Awesome.

N.B. The numbers in this post come from different corpus sizes and settings, so compare them only within the same paragraph.

Maybe adding _more_ documents? I started adding random docs from Wikipedia, "distractors", but it didn't work (corpus, in the meantime, became bigger): Wikipedia articles were disjoint from other articles so they did not distract the query system from the actual query.

Alright then ... ? Then I implemented a noise builder, a tool to pick the Wikipedia articles whose semantic distance was the lowest to the gold source and add them to the corpus. It worked! Adding 156 neighbour articles dropped MRR from 0.828 to 0.758 while recall held. The gold sources were still found, but ranked lower.

Phew, I did it! In the end, the corpus and the eval arrived where they are now

- doc size: 1283
- queries size: 84

and a small subset of it (111 docs with the most difficult 18 queries) for smoke tests, which I called wp111.

Do you remember how much time it took to index my notes? 11 hours with one LLM, 50 for the other one. Well, with these big Wikipedia articles the music changed dramatically. The Gemma model contextualization required approximately one night for 10 docs, while the Qwen model required "only" ten times less. Yes, you read it well: it took me almost one week to ingest the test corpus with the fastest model, with my poor Mac crunching night and day, fans at full speed as if there was a helicopter in my office. Don't blame me for having "too few" queries and docs please!


## Contextualization disappointment but wait ...

Finally, I was able to run the first evals and ... it turned out that contextualization was adding _nothing_ to the RAG performance. How was it possible? Was all that compute time wasted?
The contextualization analysis was scaring me because I started to think I would have to reindex the corpus several times, read-my-lips: reindex-the-corpus-several-times!

Fortunately, there were other parts to implement in my roadmap to distract myself from the depressing contextualization results.

The most promising next step was adding a reranker. A reranker would have improved Minerva's performance and given me a boost that could make contextualization useless. And so I rushed to add a reranker but, to my disappointment, the performance was still disappointing: the reranker gave some gain, less than promised, with a few strange regressions.

"Fortunately" another problem distracted me from this *double* disappointment! Upon careful inspection I found several flaws in the keyword search as well.

First problem: keyword search does not play well with natural language queries, I had to use an 'OR' over all the terms of the query.
Second: my fts settings were language neutral (that's because I wanted to index docs in any language), so no [stemming](https://en.wikipedia.org/wiki/Stemming), no [stop words](https://en.wikipedia.org/wiki/Stop_word) removal, so that 'OR' above brought in a lot of noise.

In this period I was reading some PostgreSQL docs and I wanted to run an experiment writing some Python throw-away code; soon the experiment became an exploration of full text search capabilities.

I wanted to overcome the two problems found in the full text search leg and I tried different strategies.

I tried to use the builtin fts search (the same I was using in Minerva) but with trigram matching to 'simulate' stemming, but I still had that noise problem from the 'OR' and the stop words: I really needed [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

So, I tried a real BM25 engine (ParadeDB `pg_search` was my choice). Using BM25 means using a [bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model) and thanks to IDF it was able to ignore all the frequent-but-irrelevant terms. This solved the 'OR' + stop words problem but not the stemming one.

To overcome it, I also tried fuzzy matching search with BM25 (trigram indexes didn't fit well in Minerva ingestion pipeline), but short common words matched many variants, so the results were filled with noise and I decided to live without it.

Fortunately, a consideration gave me some relief: wasn't BM25 used mostly to detect keywords and acronyms that are not caught by the semantic engine? Then the stemming problem is not that important.

Given these premises, I felt ready for a new round of changes in Minerva.

I switched from `ts_rank` to a real BM25 with the same ParadeDB `pg_search` I discovered in my previous exploration, and the FTS leg of my hybrid search improved considerably: at alpha = 0.3, that is when the weight of full text search is higher, R@5 went from 0.71 to 0.87 and MRR@10 from 0.60 to 0.75. Well done!

Then, back to the reranker! But while I was investigating the reranker, another detail got my attention: those contextualizations were so strange...

## Interlude: what do I want Minerva to become?

During the July pause, I read [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) (Gao et al., 2023) and started thinking about how many things I would have to consider adding to Minerva in the future.

If you read that article, you will find out that many advanced techniques described there were needed to build an autonomous system, with its pipeline. Query rewriting, multi-hop retrieval, and other techniques had all the reasons to be present in that scenario.

And so I realized that I had used Minerva through the MCP server for months, without ever drawing the right consequences: many of the above techniques were provided by the agent using Minerva: the agent rewrites the query, searches again and follows leads across documents by itself; I did not need to build that into Minerva.

I'm not the only guy thinking this is wise: PostHog built their own AI agent and [wished they had started with an MCP server](https://newsletter.posthog.com/p/what-we-wish-we-knew-before-building).

Alright, so what is Minerva: it is a local RAG system meant to be used as an MCP server. It provides a base high quality search capability and other composable tools (like the full document retrieval) but the pipeline should be decided by another AI based application, something like "your-favourite-foundation-model-here Desktop" or even a thinking model loaded locally with LM Studio.

## Contextualization wars

... alright, it was late August 2026 and I was looking at contextualization records and I found that Qwen ones were complete crap: wrong in the best cases, hallucinations vomiting repeated words in the worst: Qwen invented facts in the prefixes. Of 134,814 prefixes, 2,698 were longer than 2,000 characters and 207 longer than 40,000.

So, let me recap

- the fast model produced garbage
- the slow model took ages and still no improvement was visible

To reduce ingestion time, I found two very effective moves

- reduce the corpus size to speed up ingestion: half of the Wikipedia articles' text was in the references section, removed
- switch off thinking mode; observing inference on LM Studio logs, I found that the majority of time during inference was spent thinking. This was a great move: turning off thinking and using a slightly lighter Gemma 4 model (E2B, see below) allowed me to ingest the whole corpus in 42 hours

To improve contextualization I tried to attack the problem in two ways.

One was trying to find a selection of alternative LLMs. After a few tests on wp111 I found that Gemma E2B and Qwen3-1.7B scored the same (same MRR@10); I picked E2B because its contextualizations were shorter and did not repeat the section heading text (Qwen had this "header echo" in 16.5% of the samples).

The other was finding alternative ways of contextualizing chunks to see if different information improved the search performance: using header breadcrumb, using the same context for all chunks, using a complex document summary algorithm (this was an attempt to overcome the limit of small models on contextualization using the whole document I talked about earlier), trying alternative prompts. None of these gave significant improvements over not using contextualization: on the full corpus at α = 0.7, MRR@10 (mean reciprocal rank over the top 10) was the same: 0.794 without context, 0.800 with it. The worst rank of a gold source was 13 without context and 31 with it.

Some even worsened the result with respect to the no-contextualization search, for example, using the same context for all chunks meant giving the same topic words to each and making them all look alike.

In spite of all my efforts, contextualization did not help and when I looked more deeply at the problem I found that even my best selected model, Gemma, used to produce a *summary* not a contextualization; a good context does not summarize the document. *It adds what this chunk needs but does not say: the subject when the chunk does not name it, and the references it leaves unresolved*. Examples: "Michael" meaning George Michael, "this cat", "previous quarter" and so on.

In a desperate attempt to find out what a good context for a chunk would be like, I made a last attempt using heavier models with the same prompt used by the Anthropic article and I got a surprise: Claude Haiku, Claude Sonnet and Gemma 4 26B (the bulkiest local model I could load on my Mac) wrote summaries; only Gemini 3.1 Pro wrote a true context. But please note, this was a one shot test.

At this point I understood that the contextualization path was not worth it, when considering the increase of compute time vs the improvement in search, and I dropped it.

## September victories

Once I had decided that contextualization was just a dream, I focused on my last items in the roadmap to publish the project and start using it.

In a search for an easy win, I built the MCP server for Minerva2.

Then I had to face the other problem: why was the reranker not giving the advantage it promised?

It turned out that it was, again, a problem of model size. The small reranker gave unsatisfying results wrt not using it (MRR@10 rose from 0.794 to 0.841 but some gold sources fell badly, e.g. from rank 4 to 49, or from 2 to 15) but a big, fat 4B reranker improved Minerva's performance significantly. Unfortunately, it was super slow.

Latency reranking 20 candidates with bge-reranker-v2-m3 was 1.2 s, with Qwen3-Reranker-4B it was 12.1 s. But I needed to rerank a greater number of candidates, approximately 100.

So I decided to cascade them: the lean reranker pushed results up enough so that every gold source was within the first 20 ranks; then I only had to send to the second reranker a smaller set of the pool and latency became acceptable: with the cascade (bge on the top 100, then Qwen on the top 20), MRR@10 was 0.918, the worst gold rank was 5, and a query took 16.6 s.

It was clear to me that many things I read about RAGs may work but with big models and free tokens (even using Haiku for contextualization, ingesting the Wikipedia 1283 docs corpus would cost hundreds of $) but the core difficulty of Minerva2 was doing everything with local, low-performance models. It affected so many things: contextualization cost, model choice, the cascade.

At this point, the most significant features I wanted for Minerva were set. I used my evaluation pipeline to compare with version 1 and saw great improvements: the R@5 miss rate (the share of queries with no gold source in the top 5) was 7.7% for v1 and 2.4% for v2.

## Almost easy install

Finally, before publishing, I started to think about the setup needed for running Minerva:

- PostgreSQL with two extensions (pgvector and ParadeDB pg_search)
- llama.cpp running the two rerankers
- LM Studio running the document embedder
- a .NET SDK to build the binaries

well, not exactly "easy", so I made a single move to make it acceptable at least: switched to [llama-swap](https://github.com/mostlygeek/llama-swap) (a proxy that starts and stops local model servers on demand) for both rerankers and document embedder and added instructions on how to register this as a service. This meant that LM Studio was no longer needed.

I also improved configuration file management a little, deciding to adopt the [XDG](https://specifications.freedesktop.org/basedir/latest/) convention of storing them all in `~/.config/minerva/` and naming them from the binary name, while previously they all had the Microsoft standard `appsettings.json` name.

So, at least, once the installation is done, one doesn't have to open terminals and start launching cli tools: all the needed infrastructure to run Minerva MCP is a service starting at login time, and everything unloads after a timeout that I set at 15 minutes, freeing memory.

At the end of September, I uninstalled legacy Minerva, installed the new one on my Mac and started using it.

At last on 2026-10-01 I [published the project to GitHub](https://github.com/michele-orsini-71/minerva2).


## What's next?

Well, now I have to focus on other topics but sooner or later I will continue the work on this project. It was a great ride and I don't think the project is completed: there are some improvements that need to be done, others that would be nice to have.

What I think I NEED to do for a first, decent Minerva version

1. a lighter setup: these moves, for example, would leave Homebrew as the only prerequisite on a Mac
	- SQLite + sqlite-vec and FTS5 replace PostgreSQL; there won't be HNSW but for a personal archive of a few tens of thousands of chunks the linear search is still feasible
	- Ship signed and notarized binaries, so users need no .NET SDK. Publishing them self-contained would also drop the .NET Runtime dependencies.
	- Automate the model-server install with a script
2. add metadata search, filters (path, folder, date, tags) and a recency boost
3. a UI for easier admin, no more fiddling with json files

What else would be nice to have

- extract and index Claude Code conversations and Claude Desktop or, in general, any foundation model conversation for quick retrieval. I think that discussions with foundation models include lots of information one wants to search and consult from time to time
- consider other corpora for eval, e.g. [BEIR](https://github.com/beir-cellar/beir); I already did some preliminary work on it
- contextualization: have we finished? I'm not accepting it, I want to give it another try because in the meantime I had some other ideas that I want to test
- eval metrics: explore more to have other insights on RAG performance for example [Discounted cumulative gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
- more realistic queries in the evals: I could log the queries Claude actually sends and build a second dataset from them 
- multimodality: only markdown? why not use something like [MinerU](https://github.com/opendatalab/MinerU)?
- automatic re-index when notes change
- adding observability

