"""Excel data extraction models"""

from typing import List, Optional, Union, Literal
from pydantic import BaseModel, HttpUrl, Field, field_validator


class DataItem(BaseModel):
    """Represents a single data point from an Excel row"""
    item: str = Field(..., description="The metric name from Column B")
    value: Union[str, int, float] = Field(..., description="Value from Column C")
    notes: Optional[str] = Field(None, description="Context from Column D")
    source: Optional[str] = Field(None, description="Source URL from Column E")
    
    @field_validator('item')
    @classmethod
    def item_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Item name cannot be empty")
        return v.strip()
    
    @field_validator('value')
    @classmethod
    def value_not_empty(cls, v: Union[str, int, float]) -> Union[str, int, float]:
        if isinstance(v, str) and not v.strip():
            raise ValueError("Value cannot be empty string")
        return v


class DataBlock(BaseModel):
    """Represents a category block of related data items"""
    category: str = Field(..., description="Category from Column A")
    data_points: List[DataItem] = Field(default_factory=list, description="All items in this block")
    
    @field_validator('category')
    @classmethod
    def category_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Category cannot be empty")
        return v.strip()
    
    def get_item_value(self, item_name: str) -> Optional[Union[str, int, float]]:
        """Helper method to get value by item name"""
        for item in self.data_points:
            if item.item.lower() == item_name.lower():
                return item.value
        return None


class NationalLevelReport(BaseModel):
    """National level data for a country"""
    country: str
    report_type: Literal["National Level"] = "National Level"
    data_blocks: List[DataBlock] = Field(default_factory=list, description="Indices, Finance, etc.")
    
    def get_indices_block(self) -> Optional[DataBlock]:
        """Get the indices data block if it exists"""
        for block in self.data_blocks:
            if block.category.lower() == "indices":
                return block
        return None
    
    def get_block_by_category(self, category: str) -> Optional[DataBlock]:
        """Get a data block by category name"""
        for block in self.data_blocks:
            if block.category.lower() == category.lower():
                return block
        return None


class OrganizationProfile(BaseModel):
    """Profile of a soft power organization"""
    country: str
    organization_name: str
    lead_ministry: Optional[str] = None
    mission: Optional[str] = None
    budget: Optional[str] = None
    data_blocks: List[DataBlock] = Field(default_factory=list, description="All organizational data")
    
    def get_accountability_info(self) -> Optional[DataBlock]:
        """Get accountability/governance information"""
        return self.get_block_by_category("accountability")
    
    def get_finance_info(self) -> Optional[DataBlock]:
        """Get financial information"""
        return self.get_block_by_category("finance")
    
    def get_key_statistics(self) -> Optional[DataBlock]:
        """Get key statistics"""
        return self.get_block_by_category("key statistics")
    
    def get_block_by_category(self, category: str) -> Optional[DataBlock]:
        """Get a data block by category name"""
        for block in self.data_blocks:
            if block.category.lower() == category.lower():
                return block
        return None


class CountryWorkbook(BaseModel):
    """Complete data from a country Excel workbook"""
    country: str
    national_report: Optional[NationalLevelReport] = None
    organization_profiles: List[OrganizationProfile] = Field(default_factory=list)
    
    @field_validator('country')
    @classmethod
    def country_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Country name cannot be empty")
        return v.strip()
    
    def get_organization_by_name(self, name: str) -> Optional[OrganizationProfile]:
        """Find an organization by name"""
        for org in self.organization_profiles:
            if org.organization_name.lower() == name.lower():
                return org
        return None