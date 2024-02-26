# k8s-rag-operator
Kubernetes Operator to manage enterprise RAG applications based on NVIDIA services.

# Examples supported
1. [Multiturn RAG](#01-multi-turn-rag-with-memory)
2. [Multimodal RAG](#02-multi-modal-rag)
3. [CSV data based RAG ](#03-csv-based-rag)
4. [Query Decomposition RAG](#04-query-decomposition-rag)
5. [Nvidia AI Foundation based RAG](#05-nvidia-ai-foundation-rag)

## Pre-requisites

- You have deployed the canonical RAG pipeline on your client machine by following instructions from `README.md` packaged as part of this resource.

  The following pods should be running in your cluster once you have completed all the steps:
  1. NVIDIA GPU Operator Pod
  2. NVIDIA RAG Operator Pod in `rag-operator` namespace
  3. Canonical RAG pods should be running in `rag-sample` namespace as shown below
  ```console
   $ kubectl get pods -n rag-sample
   ```

   *Example Output*

   ```output
   NAME                                   READY   STATUS    RESTARTS   AGE
   frontend-7ff9c9b59c-knmk9              1/1     Running   0          53m
   nemollm-embedding-5bbc63f38d3b911f-0   1/1     Running   0          53m
   nemollm-inference-5bbc63f38d3b911f-0   1/1     Running   0          50m
   pgvector-0                             1/1     Running   0          53m
   query-router-66f8c9f6f-tvgx7           1/1     Running   0          53m
   ```

The below sections demonstrate how to deploy more examples on top of the default canonical RAG example, running in `rag-sample` namespace. By following the instuctions below, you will be able to deploy these prebuilt examples in different namespaces and talk with `pgvector`, `nemollm-embedding` and `nemollm-inference` services deployed in `rag-sample` namespace.

# 01: Multi-turn RAG with memory

## Description
This example showcases multi turn usecase in a RAG pipeline. It stores the conversation history and knowledge base in PGVector and retrieves them at runtime to understand contextual queries. It uses NeMo Inference Microservices to communicate with the embedding model and large language model.

<table class="tg">
<thead>
  <tr>
    <th class="tg-6ydv">LLM Model</th>
    <th class="tg-6ydv">Embedding</th>
    <th class="tg-6ydv">Framework</th>
    <th class="tg-6ydv">Document Type</th>
    <th class="tg-6ydv">Vector Database</th>
    <th class="tg-6ydv">Model deployment platform</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-knyo">llama-2-13b-chat</td>
    <td class="tg-knyo">nv-embed-qa</td>
    <td class="tg-knyo">Langchain Expression Language</td>
    <td class="tg-knyo">PDF/Text</td>
    <td class="tg-knyo">PGVector</td>
    <td class="tg-knyo">On Prem</td>
  </tr>
</tbody>
</table>

## Deployment

1. Pull in the helm chart from NGC
   ```
   helm fetch https://helm.ngc.nvidia.com/ohlfw0olaadg/ea-rag-examples/charts/rag-multi-turn-app-v0.4.0.tgz --username='$oauthtoken' --password=<YOUR NGC API KEY>
   ```

2. Create the example namespace
   ```
   kubectl create namespace multi-turn
   ```

3. Create the Helm pipeline instance and start the services.

   ```console
   $ helm install multi-turn rag-multi-turn-app-v0.4.0.tgz -n multi-turn --set imagePullSecret.password=<NGC_API_KEY>
   ```

4. Verify the pods are running and ready.

   ```console
   $ kubectl get pods -n multi-turn
   ```

   *Example Output*

   ```output
      NAME                                   READY   STATUS    RESTARTS   AGE
      query-router-multi-turn-5bdcd6b848-ps2ht     1/1     Running   0          74m
      rag-playground-multi-turn-6d7ff8ddf6-kgtcn   1/1     Running   0          74m

   ```

5. Access the app using port-forwarding.

   ```console
   $ kubectl port-forward service/rag-playground-multi-turn -n multi-turn 30005:8095
   ```

   Open browser and access the llm-playground UI using <http://localhost:30005>.


# 02: Multi Modal RAG

## Description
This example showcases multi modal usecase in a RAG pipeline. It can understand any kind of images in PDF (like graphs and plots) alongside text and tables. It uses multimodal models from NVIDIA AI Foundation to answer queries.

<table class="tg">
<thead>
  <tr>
    <th class="tg-6ydv">LLM Model</th>
    <th class="tg-6ydv">Embedding</th>
    <th class="tg-6ydv">Framework</th>
    <th class="tg-6ydv">Document Type</th>
    <th class="tg-6ydv">Vector Database</th>
    <th class="tg-6ydv">Model deployment platform</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-knyo">mixtral_8x7b for response generation, Deplot for graph to text convertion, Neva 22B for image to text convertion</td>
    <td class="tg-knyo">nvolveqa_40k</td>
    <td class="tg-knyo">Custom Python</td>
    <td class="tg-knyo">PDF with images</td>
    <td class="tg-knyo">Milvus</td>
    <td class="tg-knyo">Cloud - NVIDIA AI Foundation</td>
  </tr>
</tbody>
</table>

## Deployment

1. Pull in the helm chart from NGC
   ```
   helm fetch https://helm.ngc.nvidia.com/ohlfw0olaadg/ea-rag-examples/charts/rag-multimodal-app-v0.4.0.tgz --username='$oauthtoken' --password=<YOUR NGC API KEY>
   ```

3. Create the example namespace
   ```
   kubectl create namespace multimodal
   ```

4. Set the NVIDIA AI Foundation API key
   ```
   kubectl create secret -n multimodal generic nv-ai-foundation-secret --from-literal=NVIDIA_API_KEY="<NGC_API_KEY>"
   ```

5. Create the Milvus Vector DB services.

   4.1 Add the milvus repository
   ```
   helm repo add milvus https://zilliztech.github.io/milvus-helm/
   ```
   4.2 Update the helm repository
   ```
   helm repo update
   ```
   4.3 Create a file named custom_value.yaml with below content to utilize GPU's
   ```
   standalone:
     resources:
       requests:
         nvidia.com/gpu: "1"
       limits:
         nvidia.com/gpu: "1"
   ```
   4.4 Install the helm chart and point to the above created file using -f argument as shown below.
   ```
   helm install milvus milvus/milvus --set cluster.enabled=false --set etcd.replicaCount=1 --set minio.mode=standalone --set pulsar.enabled=false -f custom-values.yaml -n rag-sample
   ```

6. Create the Helm pipeline instance for core multimodal rag services.

   ```console
   $ helm install multimodal rag-multimodal-app-v0.4.0.tgz -n multimodal --set imagePullSecret.password=<NGC_API_KEY>
   ```

7. Verify the pods are running and ready.

   ```console
   $ kubectl get pods -n multimodal
   ```

   *Example Output*

   ```output
      NAME                                   READY   STATUS    RESTARTS   AGE
      query-router-multimodal-5bdcd6b848-ps2ht     1/1     Running   0          74m
      rag-playground-multimodal-6d7ff8ddf6-kgtcn   1/1     Running   0          74m
   ```

8. Access the app using port-forwarding.

   ```console
   $ kubectl port-forward service/rag-playground-multimodal -n multimodal 30004:8094
   ```

   Open browser and access the llm-playground UI using <http://localhost:30004>.


# 03: CSV based RAG

## Description
This example showcases a RAG usecase built using structured CSV data. It uses models from Nvidia AI Foundation to built the usecase. This example does not use any embedding models or vector database solution and uses PandasAI to drive the flow.

<table class="tg">
<thead>
  <tr>
    <th class="tg-6ydv">LLM Model</th>
    <th class="tg-6ydv">Embedding</th>
    <th class="tg-6ydv">Framework</th>
    <th class="tg-6ydv">Document Type</th>
    <th class="tg-6ydv">Vector Database</th>
    <th class="tg-6ydv">Model deployment platform</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-knyo">NV-Llama2-70B-RLHF</td>
    <td class="tg-knyo">Not Used</td>
    <td class="tg-knyo">PandasAI</td>
    <td class="tg-knyo">CSV</td>
    <td class="tg-knyo">Not Used</td>
    <td class="tg-knyo">Cloud - NVIDIA AI Foundation</td>
  </tr>
</tbody>
</table>

## Deployment
1. Pull in the helm chart from NGC
   ```
   helm fetch https://helm.ngc.nvidia.com/ohlfw0olaadg/ea-rag-examples/charts/rag-csv-app-v0.4.0.tgz --username='$oauthtoken' --password=<YOUR NGC API KEY>
   ```

3. Create the example namespace
   ```
   kubectl create namespace csv
   ```

4. Set the NVIDIA AI Foundation API key
   ```
   kubectl create secret -n csv generic nv-ai-foundation-secret --from-literal=NVIDIA_API_KEY="<NGC_API_KEY>"
   ```

5. Create the Helm pipeline instance and start the services.

   ```console
   $ helm install csv rag-csv-app-v0.4.0.tgz -n csv --set imagePullSecret.password=<NGC_API_KEY>
   ```

6. Verify the pods are running and ready.

   ```console
   $ kubectl get pods -n csv
   ```

   *Example Output*

   ```output
      NAME                                   READY   STATUS    RESTARTS   AGE
      query-router-csv-5bdcd6b848-ps2ht     1/1     Running   0          74m
      rag-playground-csv-6d7ff8ddf6-kgtcn   1/1     Running   0          74m

   ```

7. Access the app using port-forwarding.

   ```console
   $ kubectl port-forward service/rag-playground-csv -n csv 30003:8093
   ```

   Open browser and access the llm-playground UI using <http://localhost:30003>.


# 04: Query Decomposition RAG

## Description
This example showcases a RAG usecase built using task decomposition paradigm. It breaks down a query into smaller subtasks and then combines results from different subtasks to formulate the final answer. It uses models from Nvidia AI Foundation to built the usecase and uses Langchain as the framework.

<table class="tg">
<thead>
  <tr>
    <th class="tg-6ydv">LLM Model</th>
    <th class="tg-6ydv">Embedding</th>
    <th class="tg-6ydv">Framework</th>
    <th class="tg-6ydv">Document Type</th>
    <th class="tg-6ydv">Vector Database</th>
    <th class="tg-6ydv">Model deployment platform</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-knyo">llama-2-70b-chat</td>
    <td class="tg-knyo">nvolveqa_40k</td>
    <td class="tg-knyo">Langchain Agent</td>
    <td class="tg-knyo">PDF/Text</td>
    <td class="tg-knyo">PGVector</td>
    <td class="tg-knyo">Cloud - NVIDIA AI Foundation</td>
  </tr>
</tbody>
</table>

## Deployment

1. Pull in the helm chart from NGC
   ```
   helm fetch https://helm.ngc.nvidia.com/ohlfw0olaadg/ea-rag-examples/charts/rag-query-decomposition-app-v0.4.0.tgz --username='$oauthtoken' --password=<YOUR NGC API KEY>
   ```

3. Create the example namespace
   ```
   kubectl create namespace decompose
   ```

4. Set the NVIDIA AI Foundation API key
   ```
   kubectl create secret -n decompose generic nv-ai-foundation-secret --from-literal=NVIDIA_API_KEY="<NGC_API_KEY>"
   ```

5. Create the Helm pipeline instance and start the services.

   ```console
   $ helm install decompose rag-query-decomposition-app-v0.4.0.tgz -n decompose --set imagePullSecret.password=<NGC_API_KEY>
   ```

6. Verify the pods are running and ready.

   ```console
   $ kubectl get pods -n decompose
   ```

   *Example Output*

   ```output
      NAME                                   READY   STATUS    RESTARTS   AGE
      query-router-query-decomposition-5bdcd6b848-ps2ht     1/1     Running   0          74m
      rag-playground-query-decomposition-6d7ff8ddf6-kgtcn   1/1     Running   0          74m

   ```

7. Access the app using port-forwarding.

   ```console
   $ kubectl port-forward service/rag-playground-query-decomposition -n decompose 30002:8092
   ```

   Open browser and access the llm-playground UI using <http://localhost:30002>.


# 05: Nvidia AI Foundation RAG

## Description
This example showcases a minimilastic RAG usecase built using Nvidia AI Foundation models which can answer questions from unstructured data based documents.

<table class="tg">
<thead>
  <tr>
    <th class="tg-6ydv">LLM Model</th>
    <th class="tg-6ydv">Embedding</th>
    <th class="tg-6ydv">Framework</th>
    <th class="tg-6ydv">Document Type</th>
    <th class="tg-6ydv">Vector Database</th>
    <th class="tg-6ydv">Model deployment platform</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-knyo">mixtral_8x7b</td>
    <td class="tg-knyo">nvolveqa_40k</td>
    <td class="tg-knyo">Langchain Expression Language</td>
    <td class="tg-knyo">PDF/Text</td>
    <td class="tg-knyo">PGVector</td>
    <td class="tg-knyo">Cloud - NVIDIA AI Foundation</td>
  </tr>
</tbody>
</table>

## Deployment
1. Pull in the helm chart from NGC
   ```
   helm fetch https://helm.ngc.nvidia.com/ohlfw0olaadg/ea-rag-examples/charts/rag-nv-ai-foundation-app-v0.4.0.tgz --username='$oauthtoken' --password=<YOUR NGC API KEY>
   ```

3. Create the example namespace
   ```
   kubectl create namespace nvai
   ```

4. Set the NVIDIA AI Foundation API key
   ```
   kubectl create secret -n nvai generic nv-ai-foundation-secret --from-literal=NVIDIA_API_KEY="<NGC_API_KEY>"
   ```

5. Create the Helm pipeline instance and start the services.

   ```console
   $ helm install nvai rag-nv-ai-foundation-app-v0.4.0.tgz -n nvai --set imagePullSecret.password=<NGC_API_KEY>
   ```

6. Verify the pods are running and ready.

   ```console
   $ kubectl get pods -n nvai
   ```

   *Example Output*

   ```output
      NAME                                   READY   STATUS    RESTARTS   AGE
      query-router-ai-foundation-5bdcd6b848-ps2ht     1/1     Running   0          74m
      rag-playground-ai-foundation-6d7ff8ddf6-kgtcn   1/1     Running   0          74m

   ```

7. Access the app using port-forwarding.

   ```console
   $ kubectl port-forward service/rag-playground-ai-foundation -n nvai 30001:8091
   ```

   Open browser and access the llm-playground UI using <http://localhost:30001>.