class Like < ApplicationRecord
  belongs_to :hang_out
  belongs_to :user
  validates_uniqueness_of :hang_out_id, scope: :user_id
end
