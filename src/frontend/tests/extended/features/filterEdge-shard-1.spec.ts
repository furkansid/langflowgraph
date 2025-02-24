import { expect, test } from "@playwright/test";

test("user must see on handle click the possibility connections - RetrievalQA", async ({
  page,
}) => {
  await page.goto("/");
  await page.waitForSelector('[data-testid="mainpage_title"]', {
    timeout: 30000,
  });

  await page.waitForSelector('[id="new-project-btn"]', {
    timeout: 30000,
  });

  let modalCount = 0;
  try {
    const modalTitleElement = await page?.getByTestId("modal-title");
    if (modalTitleElement) {
      modalCount = await modalTitleElement.count();
    }
  } catch (error) {
    modalCount = 0;
  }

  while (modalCount === 0) {
    await page.getByText("New Flow", { exact: true }).click();
    await page.waitForTimeout(3000);
    modalCount = await page.getByTestId("modal-title")?.count();
  }
  await page.waitForTimeout(1000);

  await page.getByTestId(
    "input-list-plus-btn-edit_metadata_indexing_include-2",
  );

  await page.getByTestId("blank-flow").click();

  await page.waitForTimeout(1000);

  await page.getByTestId("sidebar-options-trigger").click();
  await page.getByTestId("sidebar-legacy-switch").isVisible({ timeout: 5000 });
  await page.getByTestId("sidebar-legacy-switch").click();
  await expect(page.getByTestId("sidebar-legacy-switch")).toBeChecked();
  await page.getByTestId("sidebar-options-trigger").click();

  await page.getByTestId("sidebar-search-input").click();
  await page.getByTestId("sidebar-search-input").fill("retrievalqa");

  await page.waitForTimeout(1000);
  await page
    .getByTestId("chainsRetrieval QA")
    .dragTo(page.locator('//*[@id="react-flow-id"]'));
  await page.mouse.up();
  await page.mouse.down();
  await page.getByTestId("fit_view").click();
  await page.getByTestId("zoom_out").click();
  await page.getByTestId("zoom_out").click();
  await page.getByTestId("zoom_out").click();
  await page.waitForTimeout(500);

  let visibleElementHandle;

  const outputElements = await page
    .getByTestId("handle-retrievalqa-shownode-text-right")
    .all();

  for (const element of outputElements) {
    if (await element.isVisible()) {
      visibleElementHandle = element;
      break;
    }
  }

  await visibleElementHandle.click({
    force: true,
  });

  const disclosureTestIds = [
    "disclosure-inputs",
    "disclosure-outputs",
    "disclosure-data",
    "disclosure-models",
    "disclosure-helpers",
    "disclosure-vector stores",
    "disclosure-embeddings",
    "disclosure-agents",
    "disclosure-chains",
    "disclosure-memories",
    "disclosure-prototypes",
    "disclosure-retrievers",
    "disclosure-text splitters",
  ];

  const elementTestIds = [
    "inputsChat Input",
    "outputsChat Output",
    "dataAPI Request",
    "modelsAmazon Bedrock",
    "helpersChat Memory",
    "vectorstoresAstra DB",
    "embeddingsAmazon Bedrock Embeddings",
    "agentsTool Calling Agent",
    "chainsConversationChain",
    "memoriesAstra DB Chat Memory",
    "prototypesConditional Router",
    "retrieversSelf Query Retriever",
    "textsplittersCharacterTextSplitter",
  ];

  await Promise.all(
    disclosureTestIds.map((id) => expect(page.getByTestId(id)).toBeVisible()),
  );

  await Promise.all(
    elementTestIds.map((id) =>
      expect(page.getByTestId(id).first()).toBeVisible(),
    ),
  );

  await page.getByTestId("sidebar-search-input").click();

  const visibleModelSpecsTestIds = [
    "modelsAIML",
    "modelsAmazon Bedrock",
    "modelsAnthropic",
    "modelsAzure OpenAI",
    "modelsCohere",
    "modelsGoogle Generative AI",
    "modelsGroq",
    "modelsHuggingFace",
    "modelsLM Studio",
    "modelsMaritalk",
    "modelsMistralAI",
    "modelsNVIDIA",
    "modelsOllama",
    "modelsOpenAI",
    "modelsPerplexity",
    "modelsQianfan",
    "modelsVertex AI",
  ];

  await Promise.all(
    visibleModelSpecsTestIds.map((id) =>
      expect(page.getByTestId(id)).toBeVisible(),
    ),
  );

  const chainInputElements1 = await page
    .getByTestId("handle-retrievalqa-shownode-llm-left")
    .all();

  for (const element of chainInputElements1) {
    if (await element.isVisible()) {
      visibleElementHandle = element;
      break;
    }
  }

  await visibleElementHandle.blur();

  await visibleElementHandle.click({
    force: true,
  });

  await expect(page.getByTestId("disclosure-models")).toBeVisible();

  const rqaChainInputElements0 = await page
    .getByTestId("handle-retrievalqa-shownode-template-left")
    .all();

  for (const element of rqaChainInputElements0) {
    if (await element.isVisible()) {
      visibleElementHandle = element;
      break;
    }
  }

  await visibleElementHandle.click();

  await expect(page.getByTestId("disclosure-helpers")).toBeVisible();
  await expect(page.getByTestId("disclosure-agents")).toBeVisible();
  await expect(page.getByTestId("disclosure-chains")).toBeVisible();
  await expect(page.getByTestId("disclosure-prototypes")).toBeVisible();
});
