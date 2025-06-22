# Vectorize.io Integration Setup

This financial AI platform uses Vectorize.io for managing investment factsheets, providing lightning-fast semantic search across all investment documents.

## 🚀 Quick Setup

### 1. Environment Variables

Add these to your `.env` file in the `backend/` directory:

```bash
# OpenAI Configuration (required)
OPENAI_API_KEY=your_openai_api_key_here

# Vectorize.io Configuration (required for factsheets)
VECTORIZE_API_TOKEN=your_vectorize_api_token_here
VECTORIZE_ORG_ID=your_organization_id_here
VECTORIZE_PIPELINE_ID=your_pipeline_id_here

# Optional: News API Configuration
NEWS_API_KEY=your_news_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
```

### 2. Vectorize.io Account Setup

1. **Sign up** at [vectorize.io](https://vectorize.io)
2. **Create a new index** for investment factsheets
3. **Upload your factsheets** to the index
4. **Copy your API key and index ID** to the `.env` file

### 3. Index Configuration

Configure your Vectorize.io index with these recommended settings:

```json
{
  "name": "investment-factsheets",
  "dimension": 1536,
  "metric": "cosine",
  "metadata_config": {
    "indexed": ["source", "page", "section", "fund_name", "document_type"]
  }
}
```

## 📄 Document Upload to Vectorize.io

### Recommended Metadata Structure

When uploading factsheets to Vectorize.io, use this metadata structure:

```json
{
  "source": "Fund_Name_Factsheet.pdf",
  "page": 1,
  "section": "Performance",
  "fund_name": "Example Growth Fund",
  "document_type": "factsheet",
  "text": "The actual text content from the document..."
}
```

### Document Processing Tips

1. **Chunk Size**: Keep chunks between 300-800 characters
2. **Overlap**: Use 50-100 character overlap between chunks
3. **Metadata**: Include fund names, sections, and page numbers
4. **File Naming**: Use consistent naming: `FundName_Factsheet_YYYY.pdf`

## 🔧 API Integration

### Testing Your Setup

Use this test query to verify your integration:

```bash
curl -X POST http://localhost:8000/api/orchestrated-chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What funds have the lowest expense ratios?",
    "session_id": "test-session"
  }'
```

### Available Factsheet Queries

The AI can answer questions about:

- **Performance**: Returns, volatility, benchmarks
- **Fees**: Expense ratios, management fees, load fees
- **Strategy**: Investment objectives, asset allocation
- **Risk**: Risk ratings, volatility metrics
- **Holdings**: Top holdings, sector allocation
- **Comparisons**: Side-by-side fund analysis

## 📊 Example Queries

Try these sample queries once your factsheets are uploaded:

```
"Compare the expense ratios of all equity funds"
"Show me the best performing funds over the last 5 years"
"What are the risk levels of ESG funds?"
"Find funds with exposure to technology stocks"
"Analyze the asset allocation of balanced funds"
```

## 🛠️ Troubleshooting

### Common Issues

1. **No factsheets found**: Check your `VECTORIZE_INDEX_ID`
2. **API authentication failed**: Verify your `VECTORIZE_API_KEY`
3. **Empty responses**: Ensure documents are properly indexed with metadata

### Debug Endpoints

Check these endpoints for debugging:

- `GET /api/factsheets` - List available factsheets
- `GET /api/debug/sessions` - Check session status

## 🔄 Migration from Local Storage

If you were previously using local factsheet storage:

1. ✅ **No data loss**: Local functionality remains for statements
2. ✅ **Automatic fallback**: System handles missing Vectorize.io config gracefully
3. ✅ **Enhanced performance**: Vectorize.io provides faster, more accurate search

## 📞 Support

- **Vectorize.io Docs**: [https://docs.vectorize.io](https://docs.vectorize.io)
- **API Reference**: Check the Vectorize.io API documentation
- **OpenAI Embeddings**: Using `text-embedding-3-small` model

---

**Ready to go?** Once configured, ask any question about your investment factsheets and watch the AI search across all documents instantly! 🚀 