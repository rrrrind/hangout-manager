class Comment < ApplicationRecord
  belongs_to :user
  belongs_to :hang_out
end
